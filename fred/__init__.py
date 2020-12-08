import os
import subprocess


# Always display the relevant git information
# for when this is imported into a notebook
repoPath = os.path.join(__file__, "..", "..")
repoPath = os.path.abspath(repoPath)
branch = subprocess.check_output(["git", "-C", repoPath, "branch"]).decode("utf-8")
commit = subprocess.check_output(["git", "-C", repoPath, "rev-parse", "HEAD"]).decode("utf-8")
status = "*"*20
status += "\nFred's analysis utilities\n"
status += "branch: {}commit: {}Updates:\n".format(branch, commit)
status += subprocess.check_output(["git", "-C", repoPath, "status", "-s"]).decode("utf-8")
status += "*"*20
print(status)

from . import display