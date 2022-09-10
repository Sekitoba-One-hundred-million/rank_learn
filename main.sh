dir='common'
file_list=`ls analyze`
list_file="$dir/list.txt"
write_file_name="$dir/name.py"

echo "1: name update"
echo "2: data update and learn"
echo "3: learn"

tag=""
read -p "Enter 1,2,3 > " tag

if [ !$tag = "1" ] && [ !$tag = "2" ] && [ !$tag = "3" ]; then
    echo "Wrong number"
    exit 1
fi

rm -rf $dir
mkdir $dir
echo 'class Name:' >> $write_file_name
echo '    def __init__( self ):' >> $write_file_name

for file_name in $file_list; do
    base='        self.'
    ARR=(${file_name//./ })
    name=${ARR[0]}
    #minus_name=${name}_minus
    # echo $file_name
    echo "$base$name = \"$name\"" >> $write_file_name
    echo $name >> $list_file
    #echo "$base$minus_name = \"$minus_name\"" >> $write_file_name
    done

cp -r $dir data_analyze/
cp -r $dir learn/

if [ $tag = "2" ]; then
    mpiexec -n 6 python main.py -u True
fi

if [ $tag = "3" ]; then
    python main.py
fi

rm -rf storage