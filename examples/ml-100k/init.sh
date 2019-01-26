mkdir data
cd data

wget -c http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip

echo Converting the data into a rating matrix...
julia ../convert.jl
