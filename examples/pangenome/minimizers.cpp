#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <zlib.h>

// Structure to hold a single FASTA record
struct FastaRecord {
    std::string id;
    std::string sequence;
};

// --- Function Prototypes ---

// Parses command-line arguments
void parseArguments(int argc, char* argv[], std::string& inputFile, std::string& outputFile, int& kmerLen, int& windowLen, std::string& mode);

// Reads a FASTA file (plain or gzipped) and returns a vector of FastaRecord structs
std::vector<FastaRecord> readFasta(const std::string& filename);

// Extracts minimizers for a single sequence in their order of appearance
std::vector<std::string> extractMinimizers(const std::string& sequence, int k, int w);

// --- Main Program Logic ---

int main(int argc, char* argv[]) {
    // --- Default parameters ---
    std::string inputFile = "";
    std::string outputFile = "";
    int kmerLen = 4;
    int windowLen = 5;
    std::string mode = "unique"; // Default mode

    // --- Parse command-line arguments ---
    try {
        parseArguments(argc, argv, inputFile, outputFile, kmerLen, windowLen, mode);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Usage: " << argv[0] << " --input <file> --output <file> [--kmer-len <int>] [--window-len <int>] [--mode <unique|reads|ordered>]" << std::endl;
        return 1;
    }

    // --- Check if output file already exists ---
    std::ifstream checkFile(outputFile);
    if (checkFile.good()) {
        checkFile.close();
        std::cerr << "Error: Output file '" << outputFile << "' already exists. Please use a different name or delete the existing file." << std::endl;
        return 1;
    }

    // --- Read FASTA file ---
    std::vector<FastaRecord> records;
    try {
        records = readFasta(inputFile);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error reading input file: " << e.what() << std::endl;
        return 1;
    }

    // --- Open output file ---
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open output file " << outputFile << std::endl;
        return 1;
    }

    // --- Process based on mode ---
    if (mode == "unique") {
        std::set<std::string> allUniqueMinimizers;
        for (const auto& record : records) {
            std::vector<std::string> minimizers = extractMinimizers(record.sequence, kmerLen, windowLen);
            for (const auto& m : minimizers) {
                allUniqueMinimizers.insert(m);
            }
        }
        
        // Get base name from input file path for the FASTA header
        size_t last_slash_idx = inputFile.find_last_of("\\/");
        std::string input_filename = (std::string::npos == last_slash_idx)
                                   ? inputFile
                                   : inputFile.substr(last_slash_idx + 1);

        outFile << ">" << input_filename << std::endl;

        for (const auto& m : allUniqueMinimizers) {
            outFile << m << "N" << std::endl;
        }
    } else if (mode == "reads") {
        for (const auto& record : records) {
            std::vector<std::string> extractedMinimizers = extractMinimizers(record.sequence, kmerLen, windowLen);
            // Get unique minimizers for this sequence
            std::set<std::string> uniqueMinimizers(extractedMinimizers.begin(), extractedMinimizers.end());
            
            outFile << ">" << record.id << std::endl;
            for (const auto& m : uniqueMinimizers) {
                outFile << m << "N" << std::endl;
            }
        }
    } else if (mode == "ordered") {
        for (const auto& record : records) {
            std::vector<std::string> minimizers = extractMinimizers(record.sequence, kmerLen, windowLen);
            outFile << ">" << record.id << std::endl;
            if (!minimizers.empty()) {
                // Write the first minimizer unconditionally
                outFile << minimizers[0] << "N" << std::endl;
                // Iterate from the second, only writing if it's different from the previous one
                for (size_t i = 1; i < minimizers.size(); ++i) {
                    if (minimizers[i] != minimizers[i - 1]) {
                        outFile << minimizers[i] << "N" << std::endl;
                    }
                }
            }
        }
    } else {
        std::cerr << "Error: Invalid mode specified. Use 'unique', 'reads', or 'ordered'." << std::endl;
        return 1;
    }

    std::cout << "Minimizer extraction complete. Output written to " << outputFile << std::endl;
    outFile.close();

    return 0;
}

// --- Function Implementations ---

/**
 * @brief Parses command-line arguments provided to the program.
 * @param argc The argument count.
 * @param argv The argument vector.
 * @param inputFile Reference to store the input file path.
 * @param outputFile Reference to store the output file path.
 * @param kmerLen Reference to store the k-mer length.
 * @param windowLen Reference to store the window length.
 * @param mode Reference to store the operation mode.
 */
void parseArguments(int argc, char* argv[], std::string& inputFile, std::string& outputFile, int& kmerLen, int& windowLen, std::string& mode) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input") {
            if (i + 1 < argc) {
                inputFile = argv[++i];
            } else {
                throw std::invalid_argument("--input requires a value");
            }
        } else if (arg == "--output") {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            } else {
                throw std::invalid_argument("--output requires a value");
            }
        } else if (arg == "--kmer-len") {
            if (i + 1 < argc) {
                try {
                    kmerLen = std::stoi(argv[++i]);
                } catch (const std::exception& e) {
                    throw std::invalid_argument("Invalid value for --kmer-len");
                }
            } else {
                throw std::invalid_argument("--kmer-len requires a value");
            }
        } else if (arg == "--window-len") {
            if (i + 1 < argc) {
                try {
                    windowLen = std::stoi(argv[++i]);
                } catch (const std::exception& e) {
                    throw std::invalid_argument("Invalid value for --window-len");
                }
            } else {
                throw std::invalid_argument("--window-len requires a value");
            }
        } else if (arg == "--mode") {
            if (i + 1 < argc) {
                mode = argv[++i];
            } else {
                throw std::invalid_argument("--mode requires a value");
            }
        }
    }

    if (inputFile.empty() || outputFile.empty()) {
        throw std::invalid_argument("Both --input and --output arguments are required.");
    }
}

/**
 * @brief Reads a FASTA file (plain or .gz) and parses it into a vector of records.
 * @param filename The path to the FASTA file.
 * @return A vector of FastaRecord structs.
 */
std::vector<FastaRecord> readFasta(const std::string& filename) {
    std::vector<FastaRecord> records;
    FastaRecord currentRecord;
    
    // Check if the file is gzipped
    bool is_gzipped = filename.size() > 3 && filename.substr(filename.size() - 3) == ".gz";

    auto process_line = [&](std::string& line) {
        if (line.empty()) return;

        // Trim trailing newline characters that might be left by gzgets or getline
        line.erase(line.find_last_not_of("\n\r") + 1);

        if (line[0] == '>') {
            if (!currentRecord.id.empty()) {
                records.push_back(currentRecord);
            }
            currentRecord.id = line.substr(1);
            currentRecord.sequence.clear();
        } else {
            // Append sequence, removing any potential whitespace
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            currentRecord.sequence += line;
        }
    };

    if (is_gzipped) {
        gzFile inFile = gzopen(filename.c_str(), "r");
        if (inFile == NULL) {
            throw std::runtime_error("Could not open gzipped file " + filename);
        }

        char buffer[4096]; // Read in chunks
        while (gzgets(inFile, buffer, sizeof(buffer))) {
            std::string line(buffer);
            process_line(line);
        }
        gzclose(inFile);
    } else {
        std::ifstream inFile(filename);
        if (!inFile.is_open()) {
            throw std::runtime_error("Could not open file " + filename);
        }

        std::string line;
        while (std::getline(inFile, line)) {
            process_line(line);
        }
        inFile.close();
    }

    // Add the last record
    if (!currentRecord.id.empty()) {
        records.push_back(currentRecord);
    }

    return records;
}


/**
 * @brief Gets the list of minimizers for an input sequence as they appear.
 * @param sequence The DNA/RNA sequence string.
 * @param k The size of the k-mers (minimizers).
 * @param w The size of the sliding window.
 * @return A vector of strings containing the minimizers in order of appearance.
 */
std::vector<std::string> extractMinimizers(const std::string& sequence, int k, int w) {
    std::vector<std::string> minimizers;
    if (sequence.length() < k) {
        return minimizers; // Not enough sequence for a single k-mer
    }

    // 1. Extract all k-mers from the sequence
    std::vector<std::string> kmers;
    for (size_t i = 0; i <= sequence.length() - k; ++i) {
        kmers.push_back(sequence.substr(i, k));
    }

    if (kmers.empty() || kmers.size() < w) {
        // If there are not enough k-mers to form a full window, 
        // find the single smallest k-mer in the entire sequence.
        if (!kmers.empty()) {
             minimizers.push_back(*std::min_element(kmers.begin(), kmers.end()));
        }
        return minimizers;
    }
    
    // 2. Slide a window over the k-mers and find the minimum in each window
    for (size_t i = 0; i <= kmers.size() - w; ++i) {
        // Get the current window of k-mers
        auto window_start = kmers.begin() + i;
        auto window_end = kmers.begin() + i + w;
        
        // Find the lexicographically smallest k-mer in the window
        std::string min_kmer = *std::min_element(window_start, window_end);
        minimizers.push_back(min_kmer);
    }
    
    return minimizers;
}