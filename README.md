# pg_asmjit

> An alternative JIT provider for PostgreSQL based on AsmJit. (WIP)

> [!WARNING]
> I implemented only a few operators, please don't use it!

## Requirements

- PostgreSQL 15.

## Build

```bash
git clone git@github.com:higuoxing/pg_asmjit.git
cd pg_asmjit
git submodule update --init --recursive
make PG_CONFIG=<path/to/pg_config> install
```

## Configure the `jit_provider` for PostgreSQL.

1. Edit `<path/to/postgresql.conf>` and set `jit_provider='asmjit'`.
2. Restart the PostgreSQL server.
