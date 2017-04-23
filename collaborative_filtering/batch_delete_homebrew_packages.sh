while read line
do
	brew uninstall --ignore-dependencies $line
done < /packages.ymal