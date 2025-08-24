#!/bin/sh

# dev tools > network > adventofcode.com > cookies > session
SESSION_COOKIE=""
YEAR=

day=1
while [ $day -le 25 ]; do
    curl -s -b "session=$SESSION_COOKIE" \
        "https://adventofcode.com/$YEAR/day/$day/input" \
        -o "$day.txt"
    day=$((day + 1))
done

# mv *.txt $YEAR/src
