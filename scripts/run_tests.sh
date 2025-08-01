#!/bin/bash
# Comprehensive test runner for Vietnamese Emotion Detection System

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🧪 Vietnamese Emotion Detection System - Test Suite${NC}"
echo "=================================================="

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}$1${NC}"
    echo "$(echo "$1" | sed 's/./=/g')"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup environment
print_section "🔧 Setting up test environment"

cd "$PROJECT_DIR"

# Check if pytest is installed
if ! command_exists pytest; then
    echo -e "${YELLOW}⚠️  pytest not found. Installing test dependencies...${NC}"
    pip install -r tests/requirements-test.txt
fi

# Create directories for test outputs
mkdir -p test_reports
mkdir -p htmlcov

# Function to run specific test category
run_test_category() {
    local category=$1
    local description=$2
    local marker=$3
    local extra_args=${4:-""}
    
    print_section "$description"
    
    echo "Running $category tests..."
    
    if pytest -m "$marker" tests/ $extra_args --tb=short -v; then
        echo -e "${GREEN}✅ $category tests passed${NC}"
        return 0
    else
        echo -e "${RED}❌ $category tests failed${NC}"
        return 1
    fi
}

# Initialize test results
FAILED_TESTS=()

# 1. Code Quality Checks
print_section "📝 Code Quality Checks"

echo "Running Black (code formatting)..."
if command_exists black; then
    if black --check --diff src/ api/ tests/; then
        echo -e "${GREEN}✅ Code formatting is correct${NC}"
    else
        echo -e "${YELLOW}⚠️  Code formatting issues found. Run 'black src/ api/ tests/' to fix${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Black not installed. Skipping formatting check.${NC}"
fi

echo "Running flake8 (linting)..."
if command_exists flake8; then
    if flake8 src/ api/ --max-line-length=88 --ignore=E203,W503; then
        echo -e "${GREEN}✅ No linting errors found${NC}"
    else
        echo -e "${YELLOW}⚠️  Linting issues found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  flake8 not installed. Skipping linting check.${NC}"
fi

# 2. Unit Tests
if ! run_test_category "unit" "🧩 Unit Tests" "unit" "--cov-report=html:htmlcov/unit"; then
    FAILED_TESTS+=("unit")
fi

# 3. Integration Tests  
if ! run_test_category "integration" "🔗 Integration Tests" "integration" "--cov-append"; then
    FAILED_TESTS+=("integration")
fi

# 4. API Tests
if ! run_test_category "api" "🌐 API Tests" "" "--cov-append tests/test_api.py"; then
    FAILED_TESTS+=("api")
fi

# 5. Performance Tests
print_section "⚡ Performance Tests"
echo "Running performance benchmarks..."

if pytest -m "performance" tests/ --tb=short -v --durations=10; then
    echo -e "${GREEN}✅ Performance tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  Some performance tests failed or took too long${NC}"
    FAILED_TESTS+=("performance")
fi

# 6. Security Tests
if ! run_test_category "security" "🔒 Security Tests" "security"; then
    FAILED_TESTS+=("security")
fi

# 7. Load Testing (if Artillery is available)
print_section "📊 Load Testing"

if command_exists artillery; then
    echo "Running load tests with Artillery..."
    
    # Start the API in background for load testing
    echo "Starting API server for load testing..."
    cd "$PROJECT_DIR"
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for API to start
    sleep 5
    
    # Check if API is running
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ API is running${NC}"
        
        # Run load tests if config exists
        if [ -f "load-test/artillery.yml" ]; then
            echo "Running Artillery load test..."
            if artillery run load-test/artillery.yml --output test_reports/load_test.json; then
                echo -e "${GREEN}✅ Load tests completed${NC}"
                
                # Generate HTML report
                artillery report test_reports/load_test.json --output test_reports/load_test.html
                echo "Load test report: test_reports/load_test.html"
            else
                echo -e "${YELLOW}⚠️  Load tests encountered issues${NC}"
                FAILED_TESTS+=("load")
            fi
        else
            echo -e "${YELLOW}⚠️  Artillery config not found. Skipping load tests.${NC}"
        fi
    else
        echo -e "${RED}❌ Could not start API server for load testing${NC}"
        FAILED_TESTS+=("load")
    fi
    
    # Clean up API process
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
        sleep 2
    fi
else
    echo -e "${YELLOW}⚠️  Artillery not installed. Skipping load tests.${NC}"
    echo "Install with: npm install -g artillery"
fi

# 8. Security Scan (if Bandit is available)
print_section "🛡️  Security Scan"

if command_exists bandit; then
    echo "Running Bandit security scan..."
    if bandit -r src/ api/ -f json -o test_reports/security_scan.json; then
        echo -e "${GREEN}✅ No security issues found${NC}"
    else
        echo -e "${YELLOW}⚠️  Security scan found potential issues${NC}"
        echo "Check test_reports/security_scan.json for details"
    fi
else
    echo -e "${YELLOW}⚠️  Bandit not installed. Skipping security scan.${NC}"
fi

# 9. Dependency Check
print_section "📦 Dependency Security Check"

if command_exists safety; then
    echo "Checking dependencies for security vulnerabilities..."
    if safety check --json --output test_reports/dependency_check.json; then
        echo -e "${GREEN}✅ No vulnerable dependencies found${NC}"
    else
        echo -e "${YELLOW}⚠️  Some dependencies have known vulnerabilities${NC}"
        echo "Check test_reports/dependency_check.json for details"
    fi
else
    echo -e "${YELLOW}⚠️  Safety not installed. Skipping dependency check.${NC}"
fi

# 10. Coverage Report
print_section "📊 Test Coverage Report"

echo "Generating coverage report..."
if command_exists pytest; then
    # Generate final coverage report
    coverage combine 2>/dev/null || true
    coverage report --show-missing
    coverage html -d htmlcov/
    coverage xml -o test_reports/coverage.xml
    
    echo -e "${GREEN}✅ Coverage report generated${NC}"
    echo "HTML report: htmlcov/index.html"
    echo "XML report: test_reports/coverage.xml"
fi

# 11. Model Validation Tests
print_section "Model Validation"

echo "Running model validation tests..."
python -c "
import sys, os
sys.path.append('.')

try:
    # Test model loading and basic functionality
    from src.models.lstm_model import LSTMEmotionDetector
    from src.models.baseline_models import EmotionClassifier
    
    print('Testing LSTM model...')
    lstm = LSTMEmotionDetector()
    print('✅ LSTM model instantiated successfully')
    
    print('Testing baseline models...')
    baseline = EmotionClassifier()
    print('✅ Baseline models instantiated successfully')
    
    print('✅ All models validated successfully')
    
except Exception as e:
    print(f'❌ Model validation failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Model validation passed${NC}"
else
    echo -e "${RED}❌ Model validation failed${NC}"
    FAILED_TESTS+=("model_validation")
fi

# Generate Test Summary Report
print_section "📋 Test Summary Report"

cat > test_reports/test_summary.md << EOF
# Test Summary Report

**Date**: $(date)
**Project**: Vietnamese Emotion Detection System
**Version**: 1.0.0

## Test Results

| Test Category | Status |
|---------------|--------|
| Unit Tests | $([ "${FAILED_TESTS[*]}" =~ "unit" ] && echo "❌ Failed" || echo "✅ Passed") |
| Integration Tests | $([ "${FAILED_TESTS[*]}" =~ "integration" ] && echo "❌ Failed" || echo "✅ Passed") |
| API Tests | $([ "${FAILED_TESTS[*]}" =~ "api" ] && echo "❌ Failed" || echo "✅ Passed") |
| Performance Tests | $([ "${FAILED_TESTS[*]}" =~ "performance" ] && echo "⚠️ Issues" || echo "✅ Passed") |
| Security Tests | $([ "${FAILED_TESTS[*]}" =~ "security" ] && echo "❌ Failed" || echo "✅ Passed") |
| Load Tests | $([ "${FAILED_TESTS[*]}" =~ "load" ] && echo "⚠️ Issues" || echo "✅ Passed") |
| Model Validation | $([ "${FAILED_TESTS[*]}" =~ "model_validation" ] && echo "❌ Failed" || echo "✅ Passed") |

## Coverage

$(coverage report --show-missing 2>/dev/null || echo "Coverage report not available")

## Files Generated

- HTML Coverage Report: \`htmlcov/index.html\`
- XML Coverage Report: \`test_reports/coverage.xml\`
- Load Test Report: \`test_reports/load_test.html\`
- Security Scan: \`test_reports/security_scan.json\`
- Dependency Check: \`test_reports/dependency_check.json\`

EOF

echo "Test summary report generated: test_reports/test_summary.md"

# Final Results
print_section "Final Results"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
    echo -e "${GREEN}✅ System is ready for production deployment${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed or had issues:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "${RED}  - $test${NC}"
    done
    echo -e "${YELLOW}📝 Please review the test reports and fix any issues before deployment${NC}"
    exit 1
fi 