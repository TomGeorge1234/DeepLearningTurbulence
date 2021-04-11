#!/bin/bash
git status

git add -A
a=$(date)
echo "insert comment"

read comment

echo $comment

git commit -m "$a: $comment"

git push origin master
