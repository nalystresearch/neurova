// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file layers.hpp
 * @brief Base neural network module classes
 * 
 * Neurova implementation - provides parameter management,
 * training/eval modes, and forward pass interface.
 */

#pragma once

#include "tensor.hpp"
#include <string>
#include <map>
#include <memory>
#include <vector>

namespace neurova {
namespace nn {

/**
 * @brief Base class for all neural network modules
 */
class Module : public std::enable_shared_from_this<Module> {
public:
    using ModulePtr = std::shared_ptr<Module>;
    
protected:
    std::map<std::string, Parameter> parameters_;
    std::map<std::string, ModulePtr> modules_;
    bool training_ = true;
    
public:
    Module() = default;
    virtual ~Module() = default;
    
    /**
     * @brief Forward pass - must be implemented by subclasses
     */
    virtual TensorPtr forward(const TensorPtr& x) {
        throw std::runtime_error("Subclasses must implement forward()");
    }
    
    /**
     * @brief Forward pass for multiple inputs
     */
    virtual TensorPtr forward(const TensorPtr& x1, const TensorPtr& x2) {
        throw std::runtime_error("Subclasses must implement forward() for multiple inputs");
    }
    
    /**
     * @brief Call operator - calls forward
     */
    TensorPtr operator()(const TensorPtr& x) {
        return forward(x);
    }
    
    TensorPtr operator()(const TensorPtr& x1, const TensorPtr& x2) {
        return forward(x1, x2);
    }
    
    /**
     * @brief Get all parameters
     */
    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        for (auto& [name, param] : parameters_) {
            params.push_back(&param);
        }
        for (auto& [name, module] : modules_) {
            auto sub_params = module->parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }
    
    /**
     * @brief Get named parameters
     */
    std::vector<std::pair<std::string, Parameter*>> named_parameters(const std::string& prefix = "") {
        std::vector<std::pair<std::string, Parameter*>> params;
        for (auto& [name, param] : parameters_) {
            params.emplace_back(prefix + name, &param);
        }
        for (auto& [name, module] : modules_) {
            auto sub_params = module->named_parameters(prefix + name + ".");
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }
    
    /**
     * @brief Get child modules
     */
    std::vector<ModulePtr> children() {
        std::vector<ModulePtr> result;
        for (auto& [name, module] : modules_) {
            result.push_back(module);
        }
        return result;
    }
    
    /**
     * @brief Set training mode
     */
    Module& train(bool mode = true) {
        training_ = mode;
        for (auto& [name, module] : modules_) {
            module->train(mode);
        }
        return *this;
    }
    
    /**
     * @brief Set evaluation mode
     */
    Module& eval() {
        return train(false);
    }
    
    /**
     * @brief Check if in training mode
     */
    bool is_training() const { return training_; }
    
    /**
     * @brief Zero out all parameter gradients
     */
    void zero_grad() {
        for (auto& [name, param] : parameters_) {
            param.zero_grad();
        }
        for (auto& [name, module] : modules_) {
            module->zero_grad();
        }
    }
    
    /**
     * @brief Register a parameter
     */
    void register_parameter(const std::string& name, Parameter param) {
        parameters_[name] = std::move(param);
    }
    
    /**
     * @brief Register a submodule
     */
    void register_module(const std::string& name, ModulePtr module) {
        modules_[name] = std::move(module);
    }
    
    /**
     * @brief Get parameter count
     */
    size_t num_parameters() const {
        size_t count = 0;
        for (auto& [name, param] : parameters_) {
            count += param.data->size();
        }
        for (auto& [name, module] : modules_) {
            count += module->num_parameters();
        }
        return count;
    }
};

using ModulePtr = Module::ModulePtr;

/**
 * @brief Sequential container - layers are applied in order
 */
class Sequential : public Module {
private:
    std::vector<ModulePtr> layers_;
    
public:
    Sequential() = default;
    
    Sequential(std::initializer_list<ModulePtr> layers) {
        size_t idx = 0;
        for (auto& layer : layers) {
            add_module(std::to_string(idx++), layer);
        }
    }
    
    void add_module(const std::string& name, ModulePtr module) {
        layers_.push_back(module);
        register_module(name, module);
    }
    
    void add(ModulePtr module) {
        add_module(std::to_string(layers_.size()), module);
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        TensorPtr out = x;
        for (auto& layer : layers_) {
            out = layer->forward(out);
        }
        return out;
    }
    
    ModulePtr operator[](size_t idx) {
        return layers_.at(idx);
    }
    
    size_t size() const { return layers_.size(); }
};

/**
 * @brief ModuleList - holds submodules in a list
 */
class ModuleList : public Module {
private:
    std::vector<ModulePtr> modules_list_;
    
public:
    ModuleList() = default;
    
    ModuleList(std::initializer_list<ModulePtr> modules) {
        size_t idx = 0;
        for (auto& m : modules) {
            append(m);
        }
    }
    
    void append(ModulePtr module) {
        register_module(std::to_string(modules_list_.size()), module);
        modules_list_.push_back(module);
    }
    
    ModulePtr operator[](size_t idx) {
        return modules_list_.at(idx);
    }
    
    size_t size() const { return modules_list_.size(); }
    
    auto begin() { return modules_list_.begin(); }
    auto end() { return modules_list_.end(); }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("ModuleList does not implement forward()");
    }
};

/**
 * @brief ModuleDict - holds submodules in a dictionary
 */
class ModuleDict : public Module {
private:
    std::map<std::string, ModulePtr> modules_dict_;
    
public:
    ModuleDict() = default;
    
    void insert(const std::string& key, ModulePtr module) {
        register_module(key, module);
        modules_dict_[key] = module;
    }
    
    ModulePtr operator[](const std::string& key) {
        return modules_dict_.at(key);
    }
    
    bool contains(const std::string& key) const {
        return modules_dict_.count(key) > 0;
    }
    
    size_t size() const { return modules_dict_.size(); }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("ModuleDict does not implement forward()");
    }
};

/**
 * @brief Identity - returns input unchanged
 */
class Identity : public Module {
public:
    TensorPtr forward(const TensorPtr& x) override {
        return x;
    }
};

/**
 * @brief Flatten - flatten tensor to 1D or 2D
 */
class Flatten : public Module {
private:
    int start_dim_;
    int end_dim_;
    
public:
    Flatten(int start_dim = 1, int end_dim = -1)
        : start_dim_(start_dim), end_dim_(end_dim) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto& shape = x->shape;
        int ndim = shape.size();
        
        int start = start_dim_ >= 0 ? start_dim_ : ndim + start_dim_;
        int end = end_dim_ >= 0 ? end_dim_ : ndim + end_dim_;
        
        // Compute new shape
        std::vector<size_t> new_shape;
        for (int i = 0; i < start; ++i) {
            new_shape.push_back(shape[i]);
        }
        
        size_t flat_size = 1;
        for (int i = start; i <= end; ++i) {
            flat_size *= shape[i];
        }
        new_shape.push_back(flat_size);
        
        for (int i = end + 1; i < ndim; ++i) {
            new_shape.push_back(shape[i]);
        }
        
        return x->reshape(new_shape);
    }
};

/**
 * @brief Unflatten - unflatten a tensor dimension
 */
class Unflatten : public Module {
private:
    int dim_;
    std::vector<size_t> unflattened_size_;
    
public:
    Unflatten(int dim, const std::vector<size_t>& unflattened_size)
        : dim_(dim), unflattened_size_(unflattened_size) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto& shape = x->shape;
        int ndim = shape.size();
        int dim = dim_ >= 0 ? dim_ : ndim + dim_;
        
        std::vector<size_t> new_shape;
        for (int i = 0; i < dim; ++i) {
            new_shape.push_back(shape[i]);
        }
        for (auto s : unflattened_size_) {
            new_shape.push_back(s);
        }
        for (int i = dim + 1; i < ndim; ++i) {
            new_shape.push_back(shape[i]);
        }
        
        return x->reshape(new_shape);
    }
};

} // namespace nn
} // namespace neurova
