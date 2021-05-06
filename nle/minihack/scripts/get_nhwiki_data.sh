DATA_DIR=$(python -c 'import pkg_resources; print(pkg_resources.resource_filename("nle", "minihack/dat"), end="")')

wget https://www.dropbox.com/s/6qbfmsr3l89sip0/nethackwikidata.json

cmd="mv nethackwikidata.json $DATA_DIR"
echo $cmd
$cmd
