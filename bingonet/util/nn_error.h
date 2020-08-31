
#pragma once
#include <exception>
#include <string>

namespace bingonet {

/**
 * basic exception class for Bingo-net
 **/
class nn_error : public std::exception {
public:
    explicit nn_error(const std::string& msg) : msg_(msg) {}
    const char* what() const throw() override { return msg_.c_str(); }
private:
    std::string msg_;
};

} // namespace bingonet
