-- Runs once when the local docker Postgres volume is first created.
CREATE DATABASE aitwin_test;
\c aitwin_dev
CREATE EXTENSION IF NOT EXISTS vector;
\c aitwin_test
CREATE EXTENSION IF NOT EXISTS vector;
