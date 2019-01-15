#!/bin/bash
echo "Running zip script for hw1..."
echo
echo
if [ true ];
	then
		echo "CS Login:" \(e.x.: kjin2\):
		read cslogin
		echo
		rm -f hw1_${cslogin}.zip
		zip -r hw1_${cslogin}.zip . -x "*.git*" "*data*" "*MNIST_DATA*" "*.ipynb_checkpoints*" "*zip.sh" "*requirements.txt" ".env/*" ".DS_Store"
		echo
		echo
		echo "Zip script finished, hand in with handin script."
	else
		echo "Missing required files!"
		echo
		echo "Files required:"
		echo "assignment.py"
		echo "README"
fi
