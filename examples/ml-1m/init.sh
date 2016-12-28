mkdir data
cd data

wget -c http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

echo Converting the data into a rating matrix...
julia ../convert.jl
