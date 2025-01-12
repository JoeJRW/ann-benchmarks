import subprocess
import sys

import inspect
from itertools import chain
from multiprocessing.pool import Pool
import os
import pgvector.psycopg
import psycopg

from ..base.module import BaseANN

# Environment variables
#  PERF - if "yes" then try to run "perf stat"
#  FLAMEGRAPH - if "yes" then try to run "perf record"
#  DO_START_POSTGRES - if "yes" then start Postgres, otherwise Postgres should already be running
#  POSTGRES_DB_NAME - the name of the database to use, "ann" by default
#  POSTGRES_BATCH_CONCURRENCY - number of concurrent sessions for inserts and queries.
#                               When not set the number of sessions is os.cpu_count()
#
# Options to set if MariaDB is already running
#    POSTGRES_CONN_ARGS - username, password, host, and port to connect in the format "user:password:host:port"

# TODO
#   support --batch
#   support perf stat and flamegraphs

def get_cursor(conn_args, do_autocommit):
    conn = psycopg.connect(user = conn_args["user"],
                           password = conn_args["password"],
                           dbname = conn_args["db_name"],
                           host = conn_args["host"],
                           port = conn_args["port"],
                           autocommit=do_autocommit)
    pgvector.psycopg.register_vector(conn)
    cur = conn.cursor()
    return cur

def many_queries(arg):
    cur = get_cursor(arg[0], True)
    cur.execute("SET hnsw.ef_search = %d" % arg[1])

    res = []
    for v in arg[4]:
        cur.execute(arg[2], (v, arg[3]), binary=True, prepare=True)
        res.append([id for id, in cur.fetchall()])
    cur.close()
    return res

class PGVectorBase(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self._size = 0

        # TODO - is there a cleaner way to do this?
        #self._batch = inspect.stack()[2].frame.f_locals['batch']
        #print(f"--batch is %s" % self._batch)
        print(f"%d M, %d efConstruction, %s metric" % (self._m, self._ef_construction, self._metric))

        self._batch_concurrency = int(os.environ.get('POSTGRES_BATCH_CONCURRENCY', str(os.cpu_count())))
        print(f"Postgres concurrency for --batch is %d" % self._batch_concurrency)

        self._do_start_postgres = os.environ.get('DO_START_POSTGRES', 'no') == 'yes'

        # Use socket when not None, else use user, password, host and port
        self._conn_args = { "db_name":None, "socket":None, "user":None, "password":None, "host":None, "port": None }

        self._db_name = self._conn_args["db_name"] = os.environ.get('POSTGRES_DB_NAME', 'ann')
        print(f"Postgres database name is %s" % self._db_name)

        conn_str = iadb = os.environ.get('POSTGRES_CONN_ARGS', 'none')
        conn_parts = conn_str.split(":")
        if len(conn_parts) != 4:
            raise RuntimeError(f"Could not parse POSTGRES_CONN_ARGS - %s" % conn_str)
        # POSTGRES_CONN_ARGS "user:password:database:host:port"
        self._user = self._conn_args["user"] = conn_parts[0]
        self._password = self._conn_args["password"] = conn_parts[1]
        self._host = self._conn_args["host"] = conn_parts[2]
        self._port = self._conn_args["port"] = int(conn_parts[3])

    def fit(self, X):
        if self._do_start_postgres:
            subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

        cur = get_cursor(self._conn_args, True)

        self._query = self.get_query_text(X.shape[1])
        print(f"Query is: %s" % self._query)

        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        index_ddl = self.get_index_ddl(X)
        print("creating index as %s" % index_ddl)
        cur.execute(index_ddl)

        cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        self._size = cur.fetchone()[0]
        print(f"Size is %d MB" % (self._size / (1024*1024)))

        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        # print(f"ef_search %d" % ef_search)
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        #print(f"query limit %d with ef_search %d" % (n, self._ef_search))
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def batch_query(self, X, n):
        XX=[]
        concur = self._batch_concurrency
        for i in range(concur):
            XX.append((self._conn_args, self._ef_search, self._query, n, X[int(len(X)/concur*i):int(len(X)/concur*(i+1))]))
        pool = Pool(concur)
        self._res = pool.map(many_queries, XX)

    def get_batch_results(self):
        return chain(*self._res)

    def get_memory_usage(self):
        return self._size / 1024

    def done(self):
        if self._do_start_postgres:
            subprocess.run("service postgresql stop", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)


class PGVector(PGVectorBase):
    def __init__(self, metric, method_param):
        self._is_halfvec = False
        super().__init__(metric, method_param) 

    def get_index_ddl(self, X):
        if self._metric == "angular":
            return """CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
                      WITH (m = %d, ef_construction = %d)""" % (
                   self._m, self._ef_construction)
        elif self._metric == "euclidean":
            return """CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)
                      WITH (m = %d, ef_construction = %d)""" % (
                   self._m, self._ef_construction)
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    def get_query_text(self, size):
        if self._metric == "angular":
            return "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif self._metric == "euclidean":
            return "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def __str__(self):
        return f"PGVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"


class PGVector_halfvec(PGVectorBase):
    def __init__(self, metric, method_param):
        self._is_halfvec = True
        super().__init__(metric, method_param) 

    def get_index_ddl(self, X):
        if self._metric == "angular":
            return """CREATE INDEX ON items USING hnsw ((embedding::halfvec(%d)) halfvec_cosine_ops)
                      WITH (m = %d, ef_construction = %d)""" % (
                   X.shape[1], self._m, self._ef_construction)
        elif self._metric == "euclidean":
            return """CREATE INDEX ON items USING hnsw ((embedding::halfvec(%d)) halfvec_l2_ops)
                      WITH (m = %d, ef_construction = %d)""" % (
                   X.shape[1], self._m, self._ef_construction)
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    def get_query_text(self, size):
        if self._metric == "angular":
            return "SELECT id FROM items ORDER BY embedding::halfvec(%d) <=> %%s::halfvec(%d) LIMIT %%s" % (size, size)
        elif self._metric == "euclidean":
            return "SELECT id FROM items ORDER BY embedding::halfvec(%d) <-> %%s::halfvec(%d) LIMIT %%s" % (size, size)
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def __str__(self):
        return f"PGVector_halfvec(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"

