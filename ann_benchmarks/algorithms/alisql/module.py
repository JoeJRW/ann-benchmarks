import getpass
import glob
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from multiprocessing.pool import Pool
from itertools import chain
import inspect
import numpy

import mysql.connector

from ..base.module import BaseANN

# Environment variables
#  PERF - if "yes" then try to run "perf stat"
#  FLAMEGRAPH - if "yes" then try to run "perf record"
#  DO_INIT_MYSQL - if "yes" then initialize MySQL
#  DO_START_MYSQL - if "yes" then start MySQL, otherwise MySQL should already be running
#  MYSQL_DB_NAME - the name of the database to use, "ann" by default
#  MYSQL_BATCH_CONCURRENCY - number of concurrent sessions for inserts and queries.
#                              When not set the number of sessions is os.cpu_count()
#
#  Options to set if initializing or starting MySQL here
#    MYSQL_ROOT_DIR - root directory for MySQL install
#    MYSQL_SOURCE_DIR - value for --srcdir
#    MYSQL_DB_WORKSPACE - root for database MySQL data and log directories
#
# Options to set if MySQL is already running
#    MYSQL_CONN_ARGS - username, password, host, and port to connect in the format "user:password:host:port"

def get_cursor(conn_args, do_autocommit):
    if conn_args["socket"] is not None:
        conn = mysql.connector.connect(unix_socket=conn_args["socket"],
                               database = conn_args["db_name"],
                               autocommit=do_autocommit)
    else:
        conn = mysql.connector.connect(user = conn_args["user"],
                               password = conn_args["password"],
                               host = conn_args["host"],
                               port = int(conn_args["port"]),
                               database = conn_args["db_name"],
                               autocommit=do_autocommit)
    cur = conn.cursor()
    # cur.execute("USE %s" % conn_args[1])
    return conn, cur

def many_inserts(arg):
    conn, cur = get_cursor(arg[0], False)
    cur.execute("SET rand_seed1=1, rand_seed2=2")
    rps = 100
    lenX= len(arg[2])
    start_time = time.time()
    for i, embedding in enumerate(arg[2]):
        while True:
            try:
                cur.execute("INSERT INTO t1 (id, v) VALUES (%s, %s)", (i+arg[1], vector_to_hex(embedding)))
                break
            except mysql.connector.OperationalError:
                time.sleep(0.01*(11+arg[1]*17%13))
                pass
        if arg[1] == 0 and (i + 1) % int(rps + 1) == 0:
            rps=i/(time.time()-start_time)
            print(f"{i:6d} of {lenX}, {rps:4.2f} stmt/sec, ETA {(lenX-i)/rps:.0f} sec")
    cur.execute("commit")
    cur.close()
    conn.close()

def many_queries(arg):
    conn, cur = get_cursor(arg[0], True)
    cur.execute("SET vidx_hnsw_ef_search = %s" % arg[1])
    res = []
    for v in arg[4]:
        cur.execute(f"SELECT id FROM t1 ORDER by vec_distance_{arg[2]}(v, %s) LIMIT %s", (vector_to_hex(v), arg[3]))
        res.append([id for id, in cur.fetchall()])
    cur.close()
    conn.close()
    return res

def vector_to_hex(v):
    return numpy.array(v, 'float32').tobytes().hex()

class AliSQL(BaseANN):

    def __init__(self, metric, method_param):
        self._test_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._metric = metric
        self._m = method_param['M']
        self._conn = None
        self._cur = None
        self._perf_proc = None
        self._perf_records = []
        self._perf_stats = []
        self._batch = inspect.stack()[2].frame.f_locals['batch'] # Yo-ho-ho!
        print(f"--batch is %s" % self._batch)

        self._engine = method_param['engine']
        self._size = 0

        if metric == "angular":
            self._metric='cosine'
        elif metric == "euclidean":
            self._metric='euclidean'
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
        print(f"--batch is %s" % self._batch)
        print(f"engine is %s" % self._engine)
        print(f"%d M, %s metric" % (self._m,  self._metric))

        self.prepare_options()
        self.initialize_db()
        self.start_db()

        self._conn, self._cur = get_cursor(self._conn_args, True)

    def prepare_options(self):
        self._perf_stat = os.environ.get('PERF', 'no') == 'yes' and AliSQL.can_run_perf()
        self._perf_record = os.environ.get('FLAMEGRAPH', 'no') == 'yes' and AliSQL.can_run_flamegraph()

        if self._perf_stat and self._perf_record:
            self._perf_stat = False
            print("\nWarning: Better not to enable both PERF and FLAMEGRAPH. Generating a flame graph only.\n")

        self._do_init_mysql = False
        self._socket_file = None
        self._mysql_init_cmd = None
        self._mysql_start_cmd = None
        self._mysql_proc = None
        # Use socket when not None, else use user, password, host and port
        self._conn_args = { "db_name":None, "socket":None, "user":None, "password":None, "host":None, "port": None }

        self._db_name = self._conn_args["db_name"] = os.environ.get('MYSQL_DB_NAME', 'ann')
        print(f"MySQL database name is %s" % self._db_name)

        self._batch_concurrency = int(os.environ.get('MYSQL_BATCH_CONCURRENCY', str(os.cpu_count())))
        print(f"MySQL concurrency for --batch is %d" % self._batch_concurrency)

        self._do_start_mysql = os.environ.get('DO_START_MYSQL', 'no') == 'yes'
        if not self._do_start_mysql:
            conn_str = iadb = os.environ.get('MYSQL_CONN_ARGS', 'none')
            conn_parts = conn_str.split(":")
            if len(conn_parts) != 4:
                raise RuntimeError(f"Could not parse MYSQL_CONN_ARGS - %s" % conn_str)
            # MYSQL_CONN_ARGS "user:password:database:host:port"
            self._user = self._conn_args["user"] = conn_parts[0]
            self._password = self._conn_args["password"] = conn_parts[1]
            self._host = self._conn_args["host"] = conn_parts[2]
            self._port = self._conn_args["port"] = conn_parts[3]
            return

        # MySQL build dir or installed dir
        mysql_root_dir = os.environ.get('MYSQL_ROOT_DIR')
        if mysql_root_dir is None:
            print("MySQL path MYSQL_ROOT_DIR is not provided. It can be your local build dir or installation dir. ")
            raise RuntimeError(f"Could not initialize database.")

        # Initialize database when running locally
        self._do_init_mysql = os.environ.get('DO_INIT_MYSQL', 'no') == 'yes'

        # DB workfolder: data + error log
        mysql_db_workspace = os.environ.get('MYSQL_DB_WORKSPACE')
        if mysql_db_workspace is None:
            raise RuntimeError("Please specify path MYSQL_DB_WORKSPACE to define the database directory.")
        data_dir = mysql_db_workspace + '/data'
        log_file = mysql_db_workspace + '/master-error.log'
        # Create data directory if not exist
        os.makedirs(f"{data_dir}", exist_ok=True)

        # Generate a socket file name under /tmp to make sure the file path is always under 107 character, to avoid "The socket file path is too long" error
        dset=inspect.stack()[3].frame.f_locals['dataset_name'] + '-'
        self._socket_file = self._conn_args["socket"] = tempfile.mktemp(prefix=dset, suffix='.sock', dir='/tmp')

        print("\nSetup paths:")
        print(f"MYSQL_ROOT_DIR: {mysql_root_dir}")
        print(f"DATA_DIR: {data_dir}")
        print(f"LOG_FILE: {log_file}")
        print(f"SOCKET_FILE: {self._socket_file}\n")

        # Command for MySQL initialization
        self._mysql_init_cmd = [
            glob.glob(f"{mysql_root_dir}/bin/mysqld")[0],
            "--initialize-insecure",
            "--user=mysql",
            "--default_authentication_plugin=mysql_native_password",
            "--lower_case_table_names=1",
            f"--datadir={data_dir}",
            f"--basedir={mysql_root_dir}"
        ]

        # Command for starting MySQL server
        self._mysql_start_cmd = [
            #'perf','record','-g', '--user-callchains', '--timestamp-filename', '--output=perf.perf',
            #'rr','record','-h',
            glob.glob(f"{mysql_root_dir}/bin/mysqld")[0],
            "--no-defaults",
            f"--datadir={data_dir}",
            f"--log_error={log_file}",
            f"--socket={self._socket_file}",
            "--loose-innodb-buffer-pool-size=16G",
            "--loose-vidx-disabled=false"
            "--loose-vidx-hnsw-cache-size=10G"
        ]
        user_option = AliSQL.get_user_option()
        if user_option is not None:
            self._mysql_start_cmd += user_option

    def initialize_db(self):
        if not self._do_init_mysql:
            print('Not initializing MySQL')
            return
    
        try:
            # In ann-benchmarks build, the server was initialized in Docker image, but when running locally we want to start it with a new initialization
            print("\nInitialize MySQL database...")
            print(self._mysql_init_cmd)
            #subprocess.run(self._mysql_init_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
            init_proc = subprocess.Popen(self._mysql_init_cmd, stdout=sys.stdout, stderr=sys.stderr)
            init_proc.wait()
        except Exception as e:
            print("ERROR: Failed to initialize MySQL database:", e)
            raise

    @staticmethod
    def get_user_option():
        # to support running with root user
        try:
            return ["--user=root"] if getpass.getuser() == "root" else None
        except Exception as e:
            print("Could not get current user, could be docker user mapping. Ignore.")
            return None

    @staticmethod
    def can_run_perf():
        try:
            subprocess.run(["perf", "record", "echo", "testing perf"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            print("Warning: perf command not found. Skipping.")
        except Exception as e:
            print(f"Warning: perf does not have permission to run. Skipping. Error: {e}")
        return False

    @staticmethod
    def can_run_flamegraph():
        if not AliSQL.can_run_perf():
            return False
        if not shutil.which("stackcollapse-perf.pl"):
            print("Warning: Command 'stackcollapse-perf.pl' missing. Skipping.")
            return False
        if not shutil.which("flamegraph.pl"):
            print("Warning: Command 'flamegraph.pl' missing. Skipping.")
            return False
        return True

    def perf_start(self, name):
        if not self._perf_record and not self._perf_stat:
            return

        # Clean up previous record if any
        self.perf_stop()

        if self._perf_record:
            record_name = f"perf.data.{name}.{self._test_time}"
            # perf record -p $(pidof mysqld) --freq=99 --output=perf.data.{name} --timestamp-filename"
            self._perf_proc = subprocess.Popen([
                "perf",
                "record",
                "-p",
                f"{self._mysql_proc.pid}",
                "-g",
                "--freq=100",
                "--output=results/" + record_name,
            ], stdout=sys.stdout, stderr=sys.stderr)
            self._perf_records.append(record_name)
        elif self._perf_stat:
            stat_name = f"perf.stat.{name}.{self._test_time}"
            self._perf_proc = subprocess.Popen([
                "perf",
                "stat",
                "-x,", # split by comma
                f"--output=results/{stat_name}",
                "-p",
                f"{self._mysql_proc.pid}"
            ], stdout=sys.stdout, stderr=sys.stderr)
            self._perf_stats.append(stat_name)

    def perf_stop(self):
        if (self._perf_record or self._perf_stat) and self._perf_proc is not None:
            self._perf_proc.send_signal(signal.SIGINT) # perf needs to be terminated gracefully with SIGINT
            try:
                self._perf_proc.wait(10)
                print("\nPerf process terminated.")
            except subprocess.TimeoutExpired:
                print("\nError: Perf process did not terminate within the timeout period.")
            self._perf_proc = None

    def perf_analysis(self):
        if self._perf_record:
            for record in self._perf_records:
                try:
                    flamegraph_cmd = f"perf script -i results/{record} | stackcollapse-perf.pl | flamegraph.pl > results/{record}.svg"
                    subprocess.run(flamegraph_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to generate flame graph. Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        if self._perf_stat:
            for stat_file in self._perf_stats:
                try:
                    with open(f"results/{stat_file}", 'r') as file:
                        values = [int(line.split(',')[0]) for line in file if 'cpu_core/cycles/' in line or 'cpu_atom/cycles/' in line]
                        print(f"CPU cycles in {stat_file}: {sum(values):,.0f}" if values else "Error: No CPU cycle data found.")
                except (FileNotFoundError, IOError):
                    print("Error reading the perf stat file.")

    def start_db(self):
        # Initialize database when running locally
        if not self._do_start_mysql:
            print('Not starting MySQL')
            return

        try:
            print("\nStarting MySQL server...")
            print(self._mysql_start_cmd)
            self._mysql_proc = subprocess.Popen(self._mysql_start_cmd, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            print("ERROR: Failed to start MySQL database:", e)
            raise

        # Server is expected to start in less than 30s
        start_time = time.time()
        while True:
            if time.time() - start_time > 30:
                raise TimeoutError("Timeout waiting for MySQL server to start")
            try:
                if os.path.exists(self._socket_file):
                    print("\nMySQL server started!")
                    break
            except FileNotFoundError:
                pass
            time.sleep(1)

    def fit(self, X):
        # Prepare database and table
        print("\nPreparing database and table...")
        self._cur.execute("DROP DATABASE IF EXISTS %s" % self._db_name)
        self._cur.execute("CREATE DATABASE %s" % self._db_name)
        self._cur.execute("USE %s" % self._db_name)
        ddl = f"""
          CREATE TABLE t1 (
            id INT PRIMARY KEY,
            v VECTOR({len(X[0])}) NOT NULL
          ) ENGINE={self._engine}
        """
        # VECTOR INDEX (v) M={self._m} DISTANCE={self._metric}
        print(f"ddl is: %s" % ddl)
        self._cur.execute(ddl)

        # Insert data
        print("\nInserting data...")

        start_time = time.time()
        if self._batch:
            XX=[]
            concur = self._batch_concurrency
            for i in range(concur):
                n = int((len(X) / concur) * i)
                XX.append((self._conn_args, n, X[n:int((len(X)/ concur) * (i+1))]))
            pool = Pool(concur)
            self._res = pool.map(many_inserts, XX)
        else:
            self.perf_start("inserting")
            # print(f"at insert ef_search %d" % self._ef_search)
            iconn, icur = get_cursor(self._conn_args, False)
            icur.execute("SET rand_seed1=1, rand_seed2=2")
            rps, rows, last, total=1000, 0, time.time(), 1
            for i, embedding in enumerate(X):
                icur.execute("INSERT INTO t1 (id, v) VALUES (%s, %s)", (i, vector_to_hex(embedding)))
                if i - rows > rps:
                    now=time.time()
                    rps=((i-rows)/(now-last) + 19*rps)/20
                    eta=(len(X)-i)/rps
                    total=now-start_time+eta
                    last, rows=now, i
                    print(f"{i:6_} of {len(X):_}, {rps:4.2f} stmt/sec ETA {eta:.0f} of {total:.0f} sec")
            icur.execute("commit")
            self.perf_stop()
        print(f"\nInsert time for {X.size:_} records: {time.time() - start_time:7.2f}")

        # Create index
        print("\nCreating index...")

        self.perf_start("indexing")
        start_time = time.time()
        if self._metric == "angular" or self._metric == "cosine":
            d = 'cosine'
        elif self._metric == "euclidean":
            d = 'euclidean'
            pass
        else:
            print(f"metric %s not supported" % self._metric)
            assert True

        # VECTOR INDEX (v) M={self._m} DISTANCE={self._metric}
        ddl = f"ALTER TABLE `t1` ADD VECTOR INDEX (v) M=%d DISTANCE=%s" % (self._m, d)
        print(f"ddl is: %s" % ddl)
        self._cur.execute(ddl)

        self.perf_stop()

        size_query = f"""select FILE_SIZE from information_schema.innodb_tablespaces where 
                         name = \"%s/t1__i__01\" """ % self._conn_args["db_name"]
        self._cur.execute(size_query)
        self._size = int(self._cur.fetchone()[0])
        print(f"Size is %d MB" % (self._size / (1024*1024)))

        self.perf_start("searching")

    def set_query_arguments(self, ef_search):
        # Set ef_search
        print(f"ef_search %d" % ef_search)
        self._ef_search = ef_search
        self._cur.execute("SET vidx_hnsw_ef_search = %d" % ef_search)

    def query(self, v, n):
        # print(f"query limit %d with ef_search %d" % (n, self._ef_search))
        self._cur.execute(f"SELECT id FROM t1 ORDER by vec_distance_{self._metric}(v, %s) LIMIT %s", (vector_to_hex(v), n))
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        return self._size/1024 # kB

    def batch_query(self, X, n):
        ef = self._ef_search

        XX=[]
        concur = self._batch_concurrency
        for i in range(concur):
            XX.append((self._conn_args, self._ef_search, self._metric, n, X[int(len(X)/concur*i):int(len(X)/concur*(i+1))]))
        pool = Pool()
        self._res = pool.map(many_queries, XX)

    def get_batch_results(self):
        return chain(*self._res)

    def __str__(self):
        return f"MySQL(m={self._m}, ef_search={self._ef_search})"

    def done(self):
        # Shutdown MySQL server when benchmarking done
        if self._do_start_mysql:
            self._cur.execute("shutdown")
            self._mysql_proc.wait(300)

        # Stop perf for searching and do final analysis
        self.perf_stop()
        self.perf_analysis()
