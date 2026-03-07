docker_vm_data/Ubuntu.qcow2: /tmp/osworld-cache/Ubuntu.qcow2
	mkdir -p docker_vm_data
	cp /tmp/osworld-cache/Ubuntu.qcow2 docker_vm_data/Ubuntu.qcow2

/tmp/osworld-cache/Ubuntu.qcow2: | /tmp/osworld-cache/Ubuntu.qcow2.zip
	unzip /tmp/osworld-cache/Ubuntu.qcow2.zip -d /tmp/osworld-cache

/tmp/osworld-cache/Ubuntu.qcow2.zip:
	mkdir -p /tmp/osworld-cache
	wget \
		-O /tmp/osworld-cache/Ubuntu.qcow2.zip.tmp \
		--progress=dot:giga \
		--continue \
		https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
	mv /tmp/osworld-cache/Ubuntu.qcow2.zip.tmp /tmp/osworld-cache/Ubuntu.qcow2.zip

.INTERMEDIATE: /tmp/osworld-cache/Ubuntu.qcow2.zip
