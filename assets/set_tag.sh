#!/bin/bash

# Get the current tag name
current_tag=$(git describe --abbrev=0 --tags)

# Parse the current tag and increment the version
old_version=$(echo "$current_tag" | sed -E 's/v([0-9]+\.[0-9]+\.[0-9]+)/\1/')
IFS='.' read -ra version_parts <<< "$old_version"
((version_parts[2]++))
new_version="${version_parts[0]}.${version_parts[1]}.${version_parts[2]}"
new_tag="v$new_version"

# Update the Git tag
git tag -d "$current_tag"           # Delete the old tag locally
git push origin --delete "$current_tag" # Delete the old tag on the remote repository
git tag "$new_tag"                  # Create the new tag locally
git push origin "$new_tag"          # Push the new tag to the remote repository

echo "Git tag updated from $current_tag to $new_tag"
