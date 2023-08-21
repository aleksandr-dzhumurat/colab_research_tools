#!/bin/bash

# Input code file
filename="$(pwd)/setup.py"

# Read the current version
current_version=$(grep -o "version='[0-9.]\+'" "$filename" | cut -d"'" -f2)

# Increment the version
IFS='.' read -ra version_parts <<< "$current_version"
((version_parts[2]++))
new_version="${version_parts[0]}.${version_parts[1]}.${version_parts[2]}"

# Escape slashes in the version string for sed (macOS version)
escaped_current_version=$(echo "$current_version" | sed -E 's/\//\\\//g')
escaped_new_version=$(echo "$new_version" | sed -E 's/\//\\\//g')

# Update the version in the file using sed (macOS version)
sed -i '' "s/version='$escaped_current_version'/version='$escaped_new_version'/g" "$filename"

echo "Version updated from $current_version to $new_version"