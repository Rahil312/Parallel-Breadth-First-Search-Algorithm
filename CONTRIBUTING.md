# Contributing to Parallel BFS Implementation

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the parallel BFS implementations.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Parallel-Breadth-First-Search-Algorithm.git
   cd Parallel-Breadth-First-Search-Algorithm
   ```
3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Environment

### Prerequisites
- **C/C++ Compiler**: GCC 9+ or Clang 10+
- **OpenMP**: 4.5 or later
- **MPI**: MPICH 3.3+ or OpenMPI 4.0+
- **CUDA**: 11.0+ (for GPU implementation)
- **Make**: GNU Make 4.0+

### Building the Project
```bash
make all          # Build all implementations
make clean        # Clean build artifacts
```

## 📝 Contribution Guidelines

### Code Style
- Follow existing code formatting and conventions
- Use meaningful variable names and comments
- Maintain consistent indentation (4 spaces for C++, 2 spaces for CUDA)
- Add appropriate error handling

### Performance Considerations
- Profile your changes with different graph sizes
- Ensure scalability across different core/processor counts
- Document any performance implications
- Include benchmark results when applicable

### Testing
- Test with different graph topologies (sparse, dense, scale-free)
- Verify correctness against serial BFS implementation
- Test edge cases (single vertex, disconnected graphs)
- Include performance regression tests

## 🔧 Types of Contributions

### Bug Fixes
- Include test case that reproduces the bug
- Reference any related issues
- Ensure the fix doesn't break existing functionality

### New Features
- Discuss major features in issues before implementation
- Update documentation and README
- Add appropriate tests and benchmarks
- Consider impact on all three implementations

### Optimizations
- Provide before/after performance measurements
- Explain the optimization technique used
- Document any trade-offs or limitations

### Documentation
- Improve code comments and algorithmic explanations
- Update README with new features or changes
- Add usage examples and tutorials

## 📊 Performance Benchmarking

When contributing performance improvements:

1. **Baseline Measurements**: Record performance before changes
2. **Test Environment**: Document hardware specifications
3. **Graph Datasets**: Use standard benchmark graphs when possible
4. **Metrics**: Report TEPS (Traversed Edges Per Second) and wall-clock time
5. **Scalability**: Test with different thread/process counts

## 🚦 Pull Request Process

1. **Update Documentation**: Ensure README and code comments are current
2. **Add Tests**: Include appropriate test cases
3. **Performance Verification**: Demonstrate no performance regressions
4. **Clean History**: Squash commits if necessary
5. **Descriptive PR**: Explain what, why, and how of your changes

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance optimization
- [ ] Documentation update

## Testing
- [ ] Tested on multiple graph sizes
- [ ] Verified correctness against serial implementation
- [ ] No performance regressions observed

## Performance Impact
- Baseline: X TEPS
- After changes: Y TEPS
- Improvement: Z%

## Screenshots/Benchmarks
(if applicable)
```

## 🤝 Community Guidelines

- Be respectful and constructive in discussions
- Help newcomers understand the codebase
- Share knowledge about parallel computing techniques
- Report issues clearly with reproducible examples

## 📧 Contact

For questions about contributing:
- Open an issue for bug reports or feature requests
- Start a discussion for general questions
- Tag maintainers for urgent matters

Thank you for helping improve parallel BFS implementations! 🎉