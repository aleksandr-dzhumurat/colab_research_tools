tag-manual:
	git tag v0.0.${MINOR} && git push origin v0.0.${MINOR}

tag:
	assets/set_tag.sh

make pep8:
	autopep8 --in-place --aggressive --aggressive --recursive .

version:
	assets/increment_version.sh && \
	git add setup.py