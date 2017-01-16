#!/bin/bash

# Submit script for assignments
#
# BEFORE YOU RUN THIS: check assignment 0 for the invite link,
# which will create a private submission repo linked to your GitHub username.
#
# Usage: ./submit.sh -u my_github_username

GITHUB_USERNAME="${USER}"
FORCE="false"
TARGET_BRANCH="master"

while getopts "u:f" opt; do
  case $opt in
    u)
      GITHUB_USERNAME="${OPTARG}"
      ;;
    f)
      FORCE="true"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

set -e

CHANGED=$(git diff-index --name-only HEAD --)
if [[ ! -z "$CHANGED" ]]; then
  echo "Warning! You have uncommitted changes in this repository:"
  git diff --stat
  echo "Commit before submitting? (submit.sh will only push commited changes)"
  select mode in "Yes" "No" "(cancel)"; do
    case $mode in
      "Yes" ) git commit -a; break;;
      "No" ) break;;
      "(cancel)" ) echo "Submit cancelled."; exit;;
    esac
  done
fi

REPO_NAME="datasci-w266/2017-spring-assignment-$GITHUB_USERNAME"

echo "Select preferred GitHub access protocol:"
echo "HTTPS is default, but SSH may be needed if you use two-factor auth."
select mode in "HTTPS" "SSH" "(cancel)"; do
  case $mode in
    "HTTPS" ) REMOTE_URL="https://github.com/$REPO_NAME.git"; break;;
    "SSH" ) REMOTE_URL="git@github.com:$REPO_NAME.git"; break;;
    "(cancel)" ) echo "Submit cancelled."; exit;;
  esac
done

# Set up git remote
REMOTE_ALIAS="2017-spring-assignment-submit"
echo "== Pushing to submission repo $REPO_NAME"
echo "== Latest commit: $(git rev-parse HEAD)"
echo "== Check submission status at https://github.com/$REPO_NAME"
if [[ $(git remote | grep "$REMOTE_ALIAS") ]]; then
  git remote -v remove "$REMOTE_ALIAS"
fi
git remote -v add "$REMOTE_ALIAS" "${REMOTE_URL}"
# git push "$REMOTE_ALIAS" --all
if [[ ${FORCE} == true ]]; then
  echo "Warning! Force-submit will overwrite remote history. Proceed?"
  select mode in "Force submit" "(cancel)"; do
    case $mode in
      "Force submit" ) echo "Proceeding with submission!"; break;;
      "(cancel)"  ) echo "Submit cancelled."; exit;;
    esac
  done
  git push "$REMOTE_ALIAS" "+HEAD:${TARGET_BRANCH}"
else
  git push "$REMOTE_ALIAS" "HEAD:${TARGET_BRANCH}"
fi

# Verify submission succeeded
echo "=== Verifying submission ==="
git fetch "$REMOTE_ALIAS"
if [[ $(git rev-parse HEAD) == $(git ls-remote "$REMOTE_ALIAS" "${TARGET_BRANCH}" | cut -f1) ]]; then
  echo "=== Submission successful! ==="
else
  echo "=== ERROR: Submission failed. Manually push to $REPO_NAME:${TARGET_BRANCH}, or contact the course staff for help."
  echo "=== Alternatively, re-run this script in force-mode:"
  echo "=== ./submit.sh -u your-github-username -f"
fi
