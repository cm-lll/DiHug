"""Light entry: start the deadline BEFORE any numerical imports or device work."""
import time
_START=time.monotonic()
import sys

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='short-test':
        from .supervisor import short_main
        return short_main(argv[1:],_START)
    from .worker_cli import main as worker
    return worker(argv)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('FAILED: '+repr(exc),file=sys.stderr,flush=True);raise
