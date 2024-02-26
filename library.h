#ifndef LIBRARY_H
#define LIBRARY_H

#include <thread>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <iostream>

const int E_UNION = 1;
const int E_INTERSECTION = 2;
const int E_SUBTRACTION = 3;
const int E_AND = 4;
const int E_OR = 5;
const int E_XOR = 6;
const int E_XSUBTRACTION = 7;
const int UNION = 8;
const int INTERSECTION = 9;
const int SUBTRACTION = 10;
const int AND = 11;
const int OR = 12;
const int XOR = 13;
const int XSUBTRACTION = 14;

class Evaluator {
private:
    std::unique_ptr<std::map<std::string, std::set<std::string>>> high_level_elements;
    std::unique_ptr<std::vector<std::map<std::string, std::set<std::string>>>> multidimensional_high_level_elements;
    std::unique_ptr<std::map<std::string, int>> expression_operations;
    std::unique_ptr<std::map<std::string, int>> operations;
    bool is_multidimensional_hle;

    std::map<std::string, double> highlightElements(const std::string &operation, const std::map<std::string, double> &doc);

    std::set<std::string> cartesianProduct(std::vector<std::string> &tuple_high_level_element);

public:

    Evaluator();

    ~Evaluator();

    static std::pair<std::set<int>, std::set<std::string>> expressionUnion(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2);

    static std::pair<std::set<int>, std::set<std::string>> expressionIntersection(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2);

    static std::pair<std::set<int>, std::set<std::string>> expressionSubtraction(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2);

    static std::pair<std::set<int>, std::set<std::string>> expressionAnd(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2);

    static std::pair<std::set<int>, std::set<std::string>> expressionOr(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2);

    static std::pair<std::set<int>, std::set<std::string>> expressionXOr(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2);

    static std::pair<std::set<int>, std::set<std::string>> expressionXSubtraction(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2);

    static std::map<std::string, double> setUnion(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2);

    static std::map<std::string, double> setIntersection(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2);

    static std::map<std::string, double> setSubtraction(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2);

    static std::map<std::string, double> setXSubtraction(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2);

    static std::map<std::string, double> setOr(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2);

    static std::map<std::string, double> setXOr(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2);

    static std::map<std::string, double> setAnd(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2);

    void addMultidimensionalRules(const std::vector<std::map<std::string, std::set<std::string>>> &rules);

    void addOneDimensionalRules(const std::map<std::string, std::set<std::string>> &rules);

    std::map<std::string, double> evalHistogram(std::vector<std::string> &expression, const std::map<std::string, double> &doc);

    std::pair<std::set<int>, std::set<std::string>> evalExpression(std::vector<std::string> &expression, const std::map<std::string, std::set<int>> &storage);
};

class InvertedIndex {
private:
    std::unique_ptr<std::map<std::string, std::set<int>>> storage;
    std::unique_ptr<std::map<int, std::map<std::string, double>>> hists;
    unsigned int numThreads;
    Evaluator *evaluator;

    static double documentsCoincidence(const std::map<std::string, double> &doc_a, const std::map<std::string, double> &doc_b);

public:

    InvertedIndex(Evaluator *evaluator);

    ~InvertedIndex();

    Evaluator* getEvaluator();

    void addDocument(const int &id, const std::map<std::string, double> &doc);

    void addDocuments(const std::vector<std::pair<int, std::map<std::string, double>>> &docs);

    std::vector<std::pair<int, double>> retrieveByQuerySingle(const std::vector<std::string> &expression, int count = 10, bool from_end = false, double threshold = 0.001);

    std::vector<std::pair<int, double>> retrieveByQuery(const std::vector<std::string> &expression, int count = 10, bool from_end = false, double threshold = 0.001);

    std::vector<std::pair<int, double>> retrieveByHistogramSingle(const std::map<std::string, double> &doc, int count = 10, bool from_end = false, double threshold = 0.001);

    std::vector<std::pair<int, double>> retrieveByHistogram(const std::map<std::string, double> &doc, int count = 10, bool from_end = false, double threshold = 0.001);
};

#endif //LIBRARY_H
