#include "library.h"
#include <thread>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <mutex>
#include <fstream>
#include <stack>

std::map<std::string, double> Evaluator::highlightElements(const std::string &operation, const std::map<std::string, double> &doc) {
    std::set<std::string> indexes_set;
    std::map<std::string, double> resulted_hist;
    if (this->is_multidimensional_hle) {
        auto tuple_operation_str = operation.substr(1, operation.size() - 2);
        tuple_operation_str.erase(std::remove_if(tuple_operation_str.begin(), tuple_operation_str.end(), ::isspace), tuple_operation_str.end());
        std::vector<std::string> tuple_operation;
        std::stringstream ss(tuple_operation_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            tuple_operation.push_back(token);
        }
        indexes_set = this->cartesianProduct(tuple_operation);
        for (const auto &index : indexes_set) {
            auto element = doc.find(index);
            if (element != doc.end()) {
                resulted_hist[index] = element->second;
            }
        }
    } else {
        if (this->high_level_elements->find(operation) != this->high_level_elements->end()) {
            indexes_set = (*this->high_level_elements)[operation];
            for (const auto &index : indexes_set) {
                auto element = doc.find(index);
                if (element != doc.end()) {
                    resulted_hist[index] = element->second;
                }
            }
        } else {
            auto element = doc.find(operation);
            if (element != doc.end()) {
                resulted_hist[operation] = element->second;
            }
        }
    }
    return resulted_hist;
}

std::set<std::string> Evaluator::cartesianProduct(std::vector<std::string> &tuple_high_level_element) {
    std::set<std::string> result;
    if (tuple_high_level_element.size() == this->multidimensional_high_level_elements->size()) {
        std::vector<std::vector<std::string>> product;
        product.push_back(std::vector<std::string>());
        for (int i = 0; i < tuple_high_level_element.size(); i++) {
            std::vector<std::vector<std::string>> temp;
            if ((*this->multidimensional_high_level_elements)[i].find(tuple_high_level_element[i]) != (*this->multidimensional_high_level_elements)[i].end()) {
                for (const auto& val : (*this->multidimensional_high_level_elements)[i][tuple_high_level_element[i]]) {
                    for (auto vec : product) {
                        vec.push_back(val);
                        temp.push_back(vec);
                    }
                }
            } else {
                for (auto vec : product) {
                    vec.push_back(tuple_high_level_element[i]);
                    temp.push_back(vec);
                }
            }
            product = temp;
        }
        for (const auto& vec : product) {
            std::string combination;
            for (const auto & i : vec) {
                combination += i + ", ";
            }
            combination.pop_back();
            combination.pop_back();
            result.insert(combination);
        }
    }
    return result;
}

Evaluator::Evaluator() : high_level_elements(std::make_unique<std::map<std::string, std::set<std::string>>>()),
                         multidimensional_high_level_elements(std::make_unique<std::vector<std::map<std::string, std::set<std::string>>>>()),
                         is_multidimensional_hle(false),
                         expression_operations(std::make_unique<std::map<std::string, int>>()),
                         operations(std::make_unique<std::map<std::string, int>>()) {
    (*expression_operations)["+"] = E_UNION;
    (*expression_operations)["*"] = E_INTERSECTION;
    (*expression_operations)["/"] = E_SUBTRACTION;
    (*expression_operations)["&"] = E_AND;
    (*expression_operations)["|"] = E_OR;
    (*expression_operations)["#|"] = E_XOR;
    (*expression_operations)["#/"] = E_XSUBTRACTION;

    (*operations)["+"] = UNION;
    (*operations)["*"] = INTERSECTION;
    (*operations)["/"] = SUBTRACTION;
    (*operations)["&"] = AND;
    (*operations)["|"] = OR;
    (*operations)["#|"] = XOR;
    (*operations)["#/"] = XSUBTRACTION;
}

Evaluator::~Evaluator() {
    this->high_level_elements.reset();
    this->multidimensional_high_level_elements.reset();
    this->expression_operations.reset();
    this->operations.reset();
}

std::pair<std::set<int>, std::set<std::string>> Evaluator::expressionUnion(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2) {
    const auto& doc_ids_1 = arg1.first;
    const auto& keys_1 = arg1.second;
    const auto& doc_ids_2 = arg2.first;
    const auto& keys_2 = arg2.second;
    std::set<int> doc_ids_result;
    std::set<std::string> keys_result;
    std::set_union(doc_ids_1.begin(), doc_ids_1.end(), doc_ids_2.begin(), doc_ids_2.end(), std::inserter(doc_ids_result, doc_ids_result.end()));
    std::set_union(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    return std::make_pair(std::move(doc_ids_result), std::move(keys_result));
}

std::pair<std::set<int>, std::set<std::string>> Evaluator::expressionIntersection(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2) {
    const auto& doc_ids_1 = arg1.first;
    const auto& keys_1 = arg1.second;
    const auto& doc_ids_2 = arg2.first;
    const auto& keys_2 = arg2.second;
    std::set<std::string> keys_result;
    std::set<int> doc_ids_result;
    std::set_intersection(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    if (!keys_result.empty()) {
        std::set_intersection(doc_ids_1.begin(), doc_ids_1.end(), doc_ids_2.begin(), doc_ids_2.end(), std::inserter(doc_ids_result, doc_ids_result.end()));
        return std::make_pair(std::move(doc_ids_result), std::move(keys_result));
    } else {
        return std::make_pair(std::move(doc_ids_result), std::move(keys_result));
    }
}

std::pair<std::set<int>, std::set<std::string>> Evaluator::expressionSubtraction(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2) {
    const auto& doc_ids_1 = arg1.first;
    const auto& keys_1 = arg1.second;
    const auto& keys_2 = arg2.second;
    std::set<std::string> keys_result;
    std::set_difference(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    return std::make_pair(doc_ids_1, std::move(keys_result));
}

std::pair<std::set<int>, std::set<std::string>> Evaluator::expressionAnd(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2) {
    const auto& doc_ids_1 = arg1.first;
    const auto& keys_1 = arg1.second;
    const auto& doc_ids_2 = arg2.first;
    const auto& keys_2 = arg2.second;
    std::set<std::string> keys_result;
    std::set<int> doc_ids_result;
    std::set_intersection(doc_ids_1.begin(), doc_ids_1.end(), doc_ids_2.begin(), doc_ids_2.end(), std::inserter(doc_ids_result, doc_ids_result.end()));
    std::set_union(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    return std::make_pair(std::move(doc_ids_result), std::move(keys_result));
}

std::pair<std::set<int>, std::set<std::string>> Evaluator::expressionOr(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2) {
    const auto& doc_ids_1 = arg1.first;
    const auto& keys_1 = arg1.second;
    const auto& doc_ids_2 = arg2.first;
    const auto& keys_2 = arg2.second;
    std::set<std::string> keys_result;
    std::set<int> doc_ids_result;
    std::set_union(doc_ids_1.begin(), doc_ids_1.end(), doc_ids_2.begin(), doc_ids_2.end(), std::inserter(doc_ids_result, doc_ids_result.end()));
    std::set_union(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    return std::make_pair(std::move(doc_ids_result), std::move(keys_result));
}

std::pair<std::set<int>, std::set<std::string>> Evaluator::expressionXOr(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2) {
    const auto& doc_ids_1 = arg1.first;
    const auto& keys_1 = arg1.second;
    const auto& doc_ids_2 = arg2.first;
    const auto& keys_2 = arg2.second;
    std::set<std::string> keys_result;
    std::set<int> doc_ids_result;
    std::set_symmetric_difference(doc_ids_1.begin(), doc_ids_1.end(), doc_ids_2.begin(), doc_ids_2.end(), std::inserter(doc_ids_result, doc_ids_result.end()));
    std::set_union(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    return std::make_pair(std::move(doc_ids_result), std::move(keys_result));
}

std::pair<std::set<int>, std::set<std::string>> Evaluator::expressionXSubtraction(const std::pair<std::set<int>, std::set<std::string>> &arg1, const std::pair<std::set<int>, std::set<std::string>> &arg2) {
    const auto& doc_ids_1 = arg1.first;
    const auto& keys_1 = arg1.second;
    const auto& doc_ids_2 = arg2.first;
    const auto& keys_2 = arg2.second;
    std::set<std::string> keys_result;
    std::set<int> doc_ids_result;
    std::set_difference(doc_ids_1.begin(), doc_ids_1.end(), doc_ids_2.begin(), doc_ids_2.end(), std::inserter(doc_ids_result, doc_ids_result.end()));
    std::set_difference(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    return std::make_pair(std::move(doc_ids_result), std::move(keys_result));
}

std::map<std::string, double> Evaluator::setUnion(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2) {
    std::set<std::pair<std::string, double>> keys_1;
    for (const auto &entry : arg1) {
        keys_1.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_2;
    for (const auto &entry : arg2) {
        keys_2.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_result;
    std::set_union(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    std::map<std::string, double> result;
    for (const auto &entry : keys_result) {
        result[entry.first] = entry.second;
    }
    return std::move(result);
}

std::map<std::string, double> Evaluator::setIntersection(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2) {
    std::set<std::pair<std::string, double>> keys_1;
    for (const auto &entry : arg1) {
        keys_1.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_2;
    for (const auto &entry : arg2) {
        keys_2.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_result;
    std::set_intersection(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    std::map<std::string, double> result;
    for (const auto &entry : keys_result) {
        result[entry.first] = entry.second;
    }
    return std::move(result);
}

std::map<std::string, double> Evaluator::setSubtraction(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2) {
    std::set<std::pair<std::string, double>> keys_1;
    for (const auto &entry : arg1) {
        keys_1.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_2;
    for (const auto &entry : arg2) {
        keys_2.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_result;
    std::set_difference(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    std::map<std::string, double> result;
    for (const auto &entry : keys_result) {
        result[entry.first] = entry.second;
    }
    return std::move(result);
}

std::map<std::string, double> Evaluator::setXSubtraction(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2) {
    double sum = 0.0;
    for (const auto &entry : arg2) {
        sum += entry.second;
    }
    if (sum > 0) {
        return {};
    } else {
        return arg1;
    }
}

std::map<std::string, double> Evaluator::setOr(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2) {
    std::set<std::pair<std::string, double>> keys_1;
    for (const auto &entry : arg1) {
        keys_1.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_2;
    for (const auto &entry : arg2) {
        keys_2.insert(entry);
    }
    std::set<std::pair<std::string, double>> keys_result;
    std::set_union(keys_1.begin(), keys_1.end(), keys_2.begin(), keys_2.end(), std::inserter(keys_result, keys_result.end()));
    std::map<std::string, double> result;
    for (const auto &entry : keys_result) {
        result[entry.first] = entry.second;
    }
    return std::move(result);
}

std::map<std::string, double> Evaluator::setXOr(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (const auto &entry : arg1) {
        sum1 += entry.second;
    }
    for (const auto &entry : arg2) {
        sum2 += entry.second;
    }
    if (sum1 > sum2) {
        return arg1;
    } else {
        return arg2;
    }
}

std::map<std::string, double> Evaluator::setAnd(const std::map<std::string, double> &arg1, const std::map<std::string, double> &arg2) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (const auto &entry : arg1) {
        sum1 += entry.second;
    }
    for (const auto &entry : arg2) {
        sum2 += entry.second;
    }
    if (sum1 > sum2) {
        return arg2;
    } else {
        return arg1;
    }
}

void Evaluator::addMultidimensionalRules(const std::vector<std::map<std::string, std::set<std::string>>> &rules) {
    this->is_multidimensional_hle = true;
    for (const auto &rule : rules) {
        const std::map<std::string, std::set<std::string>>& new_rule(rule);
        multidimensional_high_level_elements->push_back(new_rule);
    }
}

void Evaluator::addOneDimensionalRules(const std::map<std::string, std::set<std::string>> &rules) {
    this->is_multidimensional_hle = false;
    for (const auto &pair : rules) {
        (*this->high_level_elements)[pair.first] = pair.second;
    }
}

std::map<std::string, double> Evaluator::evalHistogram(std::vector<std::string> &expression, const std::map<std::string, double> &doc) {
    std::stack<std::map<std::string, double>> stack;
    for (const auto& token : expression) {
        if (this->operations->count(token)) {
            auto pair_2 = stack.top();
            stack.pop();
            auto pair_1 = stack.top();
            stack.pop();
            switch (this->operations->at(token)) {
                case UNION: stack.push(Evaluator::setUnion(pair_1, pair_2)); break;
                case INTERSECTION: stack.push(Evaluator::setIntersection(pair_1, pair_2)); break;
                case SUBTRACTION: stack.push(Evaluator::setSubtraction(pair_1, pair_2)); break;
                case AND: stack.push(Evaluator::setAnd(pair_1, pair_2)); break;
                case OR: stack.push(Evaluator::setOr(pair_1, pair_2)); break;
                case XOR: stack.push(Evaluator::setXOr(pair_1, pair_2)); break;
                case XSUBTRACTION: stack.push(Evaluator::setXSubtraction(pair_1, pair_2)); break;
                default: return {};
            }
        } else {
            stack.push(this->highlightElements(token, doc));
        }
    }
    return stack.top();
}


std::pair<std::set<int>, std::set<std::string>> Evaluator::evalExpression(std::vector<std::string> &expression, const std::map<std::string, std::set<int>> &storage) {
        auto operation = expression.back();
        expression.pop_back();
        auto op = this->expression_operations->find(operation);
        if (op != this->expression_operations->end()) {
            auto pair_2 = this->evalExpression(expression, storage);
            auto pair_1 = this->evalExpression(expression, storage);
            std::pair<std::set<int>, std::set<std::string>> tmp;
            switch(op->second) {
                case E_UNION: return Evaluator::expressionUnion(pair_1, pair_2);
                case E_INTERSECTION: return Evaluator::expressionIntersection(pair_1, pair_2);
                case E_SUBTRACTION: return Evaluator::expressionSubtraction(pair_1, pair_2);
                case E_AND: return Evaluator::expressionAnd(pair_1, pair_2);
                case E_OR: return Evaluator::expressionOr(pair_1, pair_2);
                case E_XOR: return Evaluator::expressionXOr(pair_1, pair_2);
                case E_XSUBTRACTION: return Evaluator::expressionXSubtraction(pair_1, pair_2);
                default: return {};
            }
        } else {
            std::set<int> doc_ids;
            std::set<std::string> indexes_set;
            if (this->is_multidimensional_hle) {
                auto tuple_operation_str = operation.substr(1, operation.size() - 2);
                tuple_operation_str.erase(std::remove_if(tuple_operation_str.begin(), tuple_operation_str.end(), ::isspace), tuple_operation_str.end());
                std::vector<std::string> tuple_operation;
                std::stringstream ss(tuple_operation_str);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    tuple_operation.push_back(token);
                }
                indexes_set = this->cartesianProduct(tuple_operation);
                for (const auto &index : indexes_set) {
                    auto &doc_ids_from_index = storage.find(index)->second;
                    doc_ids.insert(doc_ids_from_index.begin(), doc_ids_from_index.end());
                }
            } else {
                if (this->high_level_elements->find(operation) != this->high_level_elements->end()) {
                    indexes_set = (*this->high_level_elements)[operation];
                    for (const auto &index : indexes_set) {
                        auto &doc_ids_from_index = storage.find(index)->second;
                        doc_ids.insert(doc_ids_from_index.begin(), doc_ids_from_index.end());
                    }
                } else {
                    indexes_set.insert(operation);
                    auto &doc_ids_from_index = storage.find(operation)->second;
                    doc_ids.insert(doc_ids_from_index.begin(), doc_ids_from_index.end());
                }
            }
            return make_pair(doc_ids, indexes_set);
        }
    }

double InvertedIndex::documentsCoincidence(const std::map<std::string, double> &doc_a, const std::map<std::string, double> &doc_b) {
    double result = 0.0;
    const std::map<std::string, double> &doc_1 = (doc_a.size() > doc_b.size()) ? doc_b : doc_a;
    const std::map<std::string, double> &doc_2 = (doc_a.size() > doc_b.size()) ? doc_a : doc_b;
    for (const auto &element : doc_1) {
        auto it = doc_2.find(element.first);
        if (it != doc_2.end()) {
            result += std::min(element.second, it->second);
        }
    }
    return result;
}

InvertedIndex::InvertedIndex(Evaluator *evaluator) : storage(std::make_unique<std::map<std::string, std::set<int>>>()),
                                                     hists(std::make_unique<std::map<int, std::map<std::string, double>>>()),
                                                     numThreads(std::thread::hardware_concurrency()),
                                                     evaluator(evaluator){}


InvertedIndex::~InvertedIndex() {
    this->storage.reset();
    this->hists.reset();
    if (this->evaluator) {
        delete this->evaluator;
    }
}

Evaluator* InvertedIndex::getEvaluator() {
    return this->evaluator;
}

void InvertedIndex::addDocument(const int &id, const std::map<std::string, double> &doc) {
    (*this->hists)[id] = doc;
    for (const auto &entry : doc) {
        (*storage)[entry.first].insert(id);
    }
}

void InvertedIndex::addDocuments(const std::vector<std::pair<int, std::map<std::string, double>>> &docs) {
    for (const auto &doc : docs) {
        (*hists)[doc.first] = doc.second;
        addDocument(doc.first, doc.second);
    }
}

std::vector<std::pair<int, double>> InvertedIndex::retrieveByQuerySingle(const std::vector<std::string> &expression, int count, bool from_end, double threshold) {
    std::vector<std::string> copied_expression(expression);
    std::set<int> docs_set = evaluator->evalExpression(copied_expression, *this->storage).first;
    std::vector<int> docs_ids(docs_set.begin(), docs_set.end());
    std::vector<std::pair<int, double>> result;
    for (const auto &id : docs_set) {
        const std::map<std::string, double> &hist = (*this->hists)[id];
        std::vector<std::string> copied_thread_expression(expression);
        const auto &result_hist = this->evaluator->evalHistogram(copied_thread_expression, hist);
        double score = 0.0;
        for (const auto &element : result_hist) {
            score += element.second;
        }
        std::pair<int, double> similarity = std::make_pair(id, score);
        if (similarity.second >= threshold) {
            result.push_back(similarity);
        }
    }
    if (from_end) {
        std::sort(result.begin(), result.end(), [](auto a, auto b){ return a.second < b.second; });
    } else {
        std::sort(result.begin(), result.end(), [](auto a, auto b){ return a.second > b.second; });
    }
    if (result.size() > count) {
        result.resize(count);
    }
    return result;
}

std::vector<std::pair<int, double>> InvertedIndex::retrieveByQuery(const std::vector<std::string> &expression, int count, bool from_end, double threshold) {
    std::vector<std::string> copied_expression(expression);
    std::set<int> docs_set = evaluator->evalExpression(copied_expression, *this->storage).first;
    std::vector<int> docs_ids(docs_set.begin(), docs_set.end());
    std::vector<std::pair<int, double>> result;
    std::mutex mtx;
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&](unsigned int thread_id) {
            for (unsigned int j = thread_id; j < docs_ids.size(); j += numThreads) {
                const std::map<std::string, double> &hist = (*this->hists)[docs_ids[j]];
                std::vector<std::string> copied_thread_expression(expression);
                const auto &result_hist = this->evaluator->evalHistogram(copied_thread_expression, hist);
                double score = 0.0;
                for (const auto &element : result_hist) {
                    score += element.second;
                }
                std::pair<int, double> similarity = std::make_pair(docs_ids[j], score);
                if (similarity.second >= threshold) {
                    std::lock_guard<std::mutex> lock(mtx);
                    result.push_back(similarity);
                }
            }
        }, i);
    }
    for (auto &thread : threads) {
        thread.join();
    }
    if (from_end) {
        std::sort(result.begin(), result.end(), [](auto a, auto b){ return a.second < b.second; });
    } else {
        std::sort(result.begin(), result.end(), [](auto a, auto b){ return a.second > b.second; });
    }
    if (result.size() > count) {
        result.resize(count);
    }
    return result;
}

std::vector<std::pair<int, double>> InvertedIndex::retrieveByHistogramSingle(const std::map<std::string, double> &doc, int count, bool from_end, double threshold) {
    std::set<int> docs_set;
    for (const auto &iterator : doc) {
        auto element_set = this->storage->find(iterator.first)->second;
        docs_set.insert(element_set.begin(), element_set.end());
    }
    std::vector<std::pair<int, double>> ranked_docs;
    for (const auto &id : docs_set) {
        auto score = InvertedIndex::documentsCoincidence(doc,(*this->hists)[id]);
        if (score > threshold) {
            ranked_docs.emplace_back(id, score);
        }
    }
    std::sort(ranked_docs.begin(), ranked_docs.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
        return a.second > b.second;
    });
    if (from_end) {
        std::vector<std::pair<int, double>> result(ranked_docs.end() - count, ranked_docs.end());
        ranked_docs = result;
    } else {
        std::vector<std::pair<int, double>> result(ranked_docs.begin(), ranked_docs.begin() + count);
        ranked_docs = result;
    }
    return ranked_docs;
}

std::vector<std::pair<int, double>> InvertedIndex::retrieveByHistogram(const std::map<std::string, double> &doc, int count, bool from_end, double threshold) {
    std::set<int> docs_set;
    for (const auto &iterator : doc) {
        auto &element_set = this->storage->find(iterator.first)->second;
        docs_set.insert(element_set.begin(), element_set.end());
    }
    std::vector<int> docs_ids(docs_set.begin(), docs_set.end());
    std::vector<std::pair<int, double>> result;
    std::mutex mtx;
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&](unsigned int thread_id) {
            for (unsigned int j = thread_id; j < docs_ids.size(); j += numThreads) {
                const std::map<std::string, double> &hist = (*this->hists)[docs_ids[j]];
                std::pair<int, double> similarity = std::make_pair(j, InvertedIndex::documentsCoincidence(doc, hist));
                if (similarity.second >= threshold) {
                    std::lock_guard<std::mutex> lock(mtx);
                    result.push_back(similarity);
                }
            }
        }, i);
    }
    for (auto &thread : threads) {
        thread.join();
    }
    if (from_end) {
        std::sort(result.begin(), result.end(), [](auto a, auto b){ return a.second < b.second; });
    } else {
        std::sort(result.begin(), result.end(), [](auto a, auto b){ return a.second > b.second; });
    }
    if (result.size() > count) {
        result.resize(count);
    }
    return result;
}

# ifdef _WIN32
#   define DLLEXPORT __declspec( dllexport )
# else
#   define DLLEXPORT
# endif

extern "C" {
    DLLEXPORT InvertedIndex* createInvertedIndex() {
        auto evaluator = new Evaluator();
        return new InvertedIndex(evaluator);
    }

    DLLEXPORT void addDocument(InvertedIndex* index, const int id, const std::map<std::string, double>& doc) {
        index->addDocument(id, doc);
    }

    DLLEXPORT void deleteInvertedIndex(InvertedIndex* index) {
        delete index;
    }

    DLLEXPORT std::vector<std::pair<int, double>>* retrieveByQuerySingle(InvertedIndex* index, std::vector<std::string>* expression, int count, bool from_end, double threshold, int* out_size) {
        auto r = index->retrieveByQuerySingle(*expression, count, from_end, threshold);
        *out_size = r.size();
        return new std::vector<std::pair<int, double>>(r);
    }

    DLLEXPORT std::vector<std::pair<int, double>>* retrieveByQuery(InvertedIndex* index, std::vector<std::string>* expression, int count, bool from_end, double threshold, int* out_size) {
        auto r = index->retrieveByQuery(*expression, count, from_end, threshold);
        *out_size = r.size();
        return new std::vector<std::pair<int, double>>(r);
    }
    DLLEXPORT std::vector<std::pair<int, double>>* retrieveByHistogramSingle(InvertedIndex* index, std::map<std::string, double>* doc, int count, bool from_end, double threshold, int* out_size) {
        auto r = index->retrieveByHistogramSingle(*doc, count, from_end, threshold);
        *out_size = r.size();
        return new std::vector<std::pair<int, double>>(r);
    }
    DLLEXPORT std::vector<std::pair<int, double>>* retrieveByHistogram(InvertedIndex* index, std::map<std::string, double>* doc, int count, bool from_end, double threshold, int* out_size) {
        auto r = index->retrieveByHistogram(*doc, count, from_end, threshold);
        *out_size = r.size();
        return new std::vector<std::pair<int, double>>(r);
    }

    DLLEXPORT void addOneDimensionalRules(InvertedIndex* index, std::vector<std::pair<std::string, std::vector<std::string>>>* rules) {
        std::map<std::string, std::set<std::string>> converted;
        for(auto pair: *rules) {
            converted.emplace(pair.first, std::set<std::string>(pair.second.begin(), pair.second.end()));
        }
        index->getEvaluator()->addOneDimensionalRules(converted);
    }

    DLLEXPORT void addMultiDimensionalRules(InvertedIndex* index, std::vector<std::pair<std::string, std::vector<std::string>>>* rules) {
        std::map<std::string, std::set<std::string>> converted;
        for(auto pair: *rules) {
            converted.emplace(pair.first, std::set<std::string>(pair.second.begin(), pair.second.end()));
        }
        index->getEvaluator()->addMultidimensionalRules(std::vector<std::map<std::string, std::set<std::string>>>(1, converted));
    }

    DLLEXPORT std::vector<std::pair<std::string, double>>* evalHistogram(InvertedIndex* index, std::vector<std::string>* expression, std::map<std::string, double>* doc, int* out_size) {
        auto r = index->getEvaluator()->evalHistogram(*expression, *doc);
        auto vec = new std::vector<std::pair<std::string, double>>(r.begin(), r.end());
        *out_size = vec->size();
        return vec;
    }

    DLLEXPORT std::pair<std::vector<int>, std::vector<std::string>>* evalExpression(InvertedIndex* index, std::vector<std::string>* expression, std::vector<std::pair<std::string, std::vector<int>>>* storage, int* size1, int* size2) {
        std::map<std::string, std::set<int>> converted;
        for(auto pair: *storage) {
            converted.emplace(pair.first, std::set<int>(pair.second.begin(), pair.second.end()));
        }
        auto r = index->getEvaluator()->evalExpression(*expression, converted);
        *size1 = r.first.size();
        *size2 = r.second.size();
        return new std::pair<std::vector<int>, std::vector<std::string>>(
                std::vector<int>(r.first.begin(), r.first.end()),
                std::vector<std::string>(r.second.begin(), r.second.end())
        );
    }

    DLLEXPORT std::map<std::string, double>* newMapStringDouble() {
        return new std::map<std::string, double>;
    }
    DLLEXPORT void deleteMapStringDouble(std::map<std::string, double>* obj) {
        delete obj;
    }
    DLLEXPORT void insertIntoMapStringDouble(std::map<std::string, double>* obj, const char* key, double value) {
        obj->emplace(std::string(key), value);
    }

    DLLEXPORT std::vector<std::string>* newVectorString() {
        return new std::vector<std::string>();
    }
    DLLEXPORT void deleteVectorString(std::vector<std::string>* obj) {
        delete obj;
    }
    DLLEXPORT void insertIntoVectorString(std::vector<std::string>* obj, const char* value) {
        obj->push_back(std::string(value));
    }

    DLLEXPORT void getFromVectorIntDouble(std::vector<std::pair<int, double>>* obj, int index, int* out_int, double* out_double) {
        *out_int = (*obj)[index].first;
        *out_double = (*obj)[index].second;
    }
    DLLEXPORT void deleteVectorIntDouble(std::vector<std::pair<int, double>>* obj) {
        delete obj;
    }
    DLLEXPORT const char* getFromVectorStringDouble(std::vector<std::pair<std::string, double>>* obj, int index, double* out_double) {
        *out_double = (*obj)[index].second;
        return (*obj)[index].first.c_str();
    }
    DLLEXPORT void deleteVectorStringDouble(std::vector<std::pair<std::string, double>>* obj) {
        delete obj;
    }
    DLLEXPORT int getFirstFromPairVectorIntVectorString(std::pair<std::vector<int>, std::vector<std::string>>* obj, int index) {
        return obj->first[index];
    }
    DLLEXPORT const char* getSecondFromPairVectorIntVectorString(std::pair<std::vector<int>, std::vector<std::string>>* obj, int index) {
        return obj->second[index].c_str();
    }
    DLLEXPORT void deletePairVectorIntVectorString(std::pair<std::vector<int>, std::vector<std::string>>* obj) {
        delete obj;
    }

    DLLEXPORT std::vector<std::pair<std::string, std::vector<std::string>>>* newVectorPairStringVectorString() {
        return new std::vector<std::pair<std::string, std::vector<std::string>>>();
    }
    DLLEXPORT void pushOuterToVectorPairStringVectorString(std::vector<std::pair<std::string, std::vector<std::string>>>* obj, const char* value) {
        obj->push_back({std::string(value), std::vector<std::string>()});
    }
    DLLEXPORT void pushInnerToVectorPairStringVectorString(std::vector<std::pair<std::string, std::vector<std::string>>>* obj, const char* value) {
        obj->back().second.push_back(std::string(value));
    }
    DLLEXPORT void deleteVectorPairStringVectorString(std::vector<std::pair<std::string, std::vector<std::string>>>* obj) {
        delete obj;
    }

    DLLEXPORT std::vector<std::pair<std::string, std::vector<int>>>* newVectorPairStringVectorInt() {
        return new std::vector<std::pair<std::string, std::vector<int>>>();
    }
    DLLEXPORT void pushOuterToVectorPairStringVectorInt(std::vector<std::pair<std::string, std::vector<int>>>* obj, const char* value) {
        obj->push_back({std::string(value), std::vector<int>()});
    }
    DLLEXPORT void pushInnerToVectorPairStringVectorInt(std::vector<std::pair<std::string, std::vector<int>>>* obj, int value) {
        obj->back().second.push_back(value);
    }
    DLLEXPORT void deleteVectorPairStringVectorInt(std::vector<std::pair<std::string, std::vector<int>>>* obj) {
        delete obj;
    }
}
