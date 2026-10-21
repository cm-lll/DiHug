"""Re-measure SAME reader cases after defensive guard; original ledger applies."""
import sys
import large
from diagnose import OUT
large.OUT=OUT/'guard_cost'
if __name__=='__main__':sys.exit(large.run(sys.argv[1]))
