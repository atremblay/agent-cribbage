import sys
from job.register import registry

if __name__ == "__main__":
    registry[sys.argv[1]]().job()
