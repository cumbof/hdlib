"""Implementation of a HD representation of de Bruijn graphs for representing pangenomes and 
classifying microbial species with hdlib."""

__author__ = ("Fabio Cumbo (fabio.cumbo@gmail.com)")

__version__ = "1.1.0"
__date__ = "Jul 26, 2025"

import os
import time
import multiprocessing as mp
from pathlib import Path

import tqdm

from hdlib.graph import Graph


# Path to the mapping genome taxonomic label
genomes_table_filepath = "genomes.tsv"

# Path to the folder with genomes minimizers fasta files (*.fna)
# Genomes are retrieved via the genomes.sh script, while the minimizers are extracted with the minimizers.cpp utility
genomes_files_folder = "genomes"

NPROC = 24
ERROR_MITIGATION_ITERS = 10

# Load the genomes table into a dictionary
# key: taxonomy, values: list of genomes
GENOMES = dict()

with open("genomes.tsv") as genomes_table:
    for line in genomes_table:
        line = line.strip()

        if line:
            # Skip the header line
            if not line.startswith("#"):
                line_split = line.split("\t")

                if line_split[1] not in GENOMES:
                    GENOMES[line_split[1]] = list()

                GENOMES[line_split[1]].append(line_split[0])

# Select genomes for training the HDC graph model
GENOMES_TRAINING = list()

for taxonomy in GENOMES:
    # Use 1 genome for the training set and the other 1 for the test set
    selection = int(len(GENOMES[taxonomy])*50/100)

    GENOMES_TRAINING.extend(GENOMES[taxonomy][:selection])

# Locate the fasta files
genomes_files = Path(genomes_files_folder).glob("*.fna")

# Extract and keep track of the minimizers
# Define edges as consecutive minimizers
# This is the set of edges used to build the graph model indexed by the taxonomic label
train_edges = set()

# This is the set of edges indexed by taxonomic label used to test the graph models
test_edges = dict()

genomes_files_dict = dict()

for filepath in genomes_files:
    genome_filename = "_".join(os.path.basename(str(filepath)).split("_")[:2])

    genomes_files_dict[genome_filename] = str(filepath)

for taxonomy in GENOMES:
    print(taxonomy)

    test_edges[taxonomy] = dict()

    for genome in GENOMES[taxonomy]:
        # Define the path to the fasta file with minimizers
        genome_path = genomes_files_dict[genome]

        if os.path.isfile(genome_path):
            print("\t{}".format(genome_path))

            with open(genome_path) as genome_file:
                for line in genome_file:
                    line = line.strip()

                    if line:
                        if line.startswith(">"):
                            prev_minimizer = None

                        else:
                            if prev_minimizer == None:
                                prev_minimizer = line[:-1]  # Remove the last N character

                            else:
                                curr_minimizer = line[:-1]
                                edge = (prev_minimizer, curr_minimizer, taxonomy)

                                if genome in GENOMES_TRAINING:
                                    train_edges.add(edge)

                                else:
                                    if genome not in test_edges[taxonomy]:
                                        test_edges[taxonomy][genome] = set()

                                    test_edges[taxonomy][genome].add(edge)
                                
                                prev_minimizer = curr_minimizer

start_time = time.time()

# Define the HD representation of a directed weighted graph
graph = Graph(size=10000, directed=True, seed=0)

print("Train edges {}".format(len(train_edges)))

# Fit the graph model (training)
graph.fit(train_edges)

print("Model built in {} seconds".format(time.time() - start_time))

if ERROR_MITIGATION_ITERS > 0:
    start_time = time.time()

    # Error mitigate the graph model
    graph.error_mitigation(max_iter=ERROR_MITIGATION_ITERS, nproc=NPROC)

    print("Error mitigation performed in {} seconds".format(time.time() - start_time))

total_test_genomes = 0

for taxonomy in test_edges:
    for genome in test_edges[taxonomy]:
        total_test_genomes += 1

start_time = time.time()

print("Genome\tReal Class\tPredicted Class\tAccuracy\tEdges")

if NPROC > 1:
    if NPROC > total_test_genomes:
        NPROC = total_test_genomes

    if NPROC > os.cpu_count()
        NPROC = os.cpu_count()

    predictions = dict()

    # Profile genomes in parallel
    with mp.Pool(processes=NPROC) as pool, tqdm.tqdm(total=total_test_genomes) as progress_bar:
        def progress(*args):
            progress_bar.update()

        jobs = [
            pool.apply_async(
                Graph._predict, 
                args=(
                    graph,
                    genome,
                    taxonomy,
                    test_edges[taxonomy][genome],
                ),
                callback=progress
            ) for taxonomy in test_edges for genome in test_edges[taxonomy]
        ]

        for job in jobs:
            genome, y_true, (y_pred, accuracy) = job.get()

            predictions[genome] = (y_true, y_pred, accuracy)

    for genome in predictions:
        y_true, y_pred, accuracy = predictions[genome]

        print("{}\t{}\t{}\t{}\t{}".format(genome, y_true, y_pred, round(accuracy, 4), len(test_edges[y_true][genome])))

else:
    for taxonomy in test_edges:
        for genome in test_edges[taxonomy]:
            genome_edges = test_edges[taxonomy][genome]

            # Real class
            y_true = taxonomy

            # Get the predicted class and its accuracy
            y_pred, accuracy = graph.predict(genome_edges)

            print("{}\t{}\t{}\t{}\t{}".format(genome, y_true, y_pred, round(accuracy, 4), len(genome_edges)))

print("Model tested in {} seconds".format(time.time() - start_time))
