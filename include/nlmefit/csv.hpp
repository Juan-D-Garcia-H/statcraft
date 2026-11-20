// SPDX-License-Identifier: Apache-2.0
//
// Minimal numeric CSV reader.
//
// The reader expects a header line followed by rows of comma-separated numbers.
// All columns are stored as double-precision vectors, preserving the column order
// declared by the header. Quoted header names are accepted and their quotes are
// stripped.
//
// The type is intentionally small and dependency-free: it is meant for loading
// example data sets, not for production-grade CSV parsing (no escaping rules,
// no locale handling and no blank-field tolerance).
#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

/// Column-oriented table loaded from a numeric CSV file.
///
/// `names[i]` is the header of column `i` and `col[i]` contains its values.
/// Access by name is provided through `operator[](const std::string&)`.
struct Csv {
    std::vector<std::string> names;
    std::vector<std::vector<double>> col;

    /// Return the column whose header is `n`.
    ///
    /// @throws std::runtime_error if no column with that name exists.
    const std::vector<double>& operator[](const std::string& n) const {
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == n) return col[i];
        throw std::runtime_error("csv: unknown column '" + n + "'");
    }
};

/// Read a numeric CSV file with a single header row.
///
/// The first line is parsed as the header; every subsequent non-empty line is
/// parsed as one row of doubles and appended to the corresponding columns.
///
/// @param path Path of the file to read.
/// @return The parsed table.
/// @throws std::runtime_error if the file cannot be opened.
inline Csv read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("csv: cannot open file '" + path + "'");

    Csv c;
    std::string line;

    std::getline(f, line);
    {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty() && token.front() == '"') token = token.substr(1, token.size() - 2);
            c.names.push_back(token);
        }
    }

    c.col.resize(c.names.size());
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        size_t i = 0;
        while (std::getline(ss, token, ',')) c.col[i++].push_back(std::stod(token));
    }
    return c;
}
