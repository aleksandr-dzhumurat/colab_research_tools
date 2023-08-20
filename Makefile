tag:
	git tag v0.0.${MINOR} && git push origin v0.0.${MINOR}

make pep8:
	autopep8 --in-place --aggressive --aggressive --recursive .