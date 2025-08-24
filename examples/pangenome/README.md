# Graph encoding and classification

This example demonstrates the use of _hdlib_'s graph functionalities to model a viral pangenome for the purpose of viral species classification.

The script in this folder performs the following steps:

1. __Data retrieval__: [fetch reference genomes](https://github.com/cumbof/hdlib/blob/main/examples/pangenome/genomes.sh) from the NCBI GenBank database with;
2. __Pangenome construction__: build a pangenome graph where nodes represent genomic segments and edges represent their adjacencies;
3. __Graph encoding__: use _hdlib.models.graph_ to encode the entire pangenome structure into a single hypervector.