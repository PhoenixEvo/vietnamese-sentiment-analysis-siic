#!/usr/bin/env python3
"""
Unified evaluation script for SIIC Vietnamese Emotion Detection System.

Usage:
    python scripts/evaluate.py --comprehensive
    python scripts/evaluate.py --model phobert
    python scripts/evaluate.py --generate-report
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser(description='Evaluate emotion detection models')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation of all models')
    parser.add_argument('--model', choices=['phobert', 'lstm', 'baselines'],
                       help='Evaluate specific model')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate final evaluation report')
    
    args = parser.parse_args()
    
    if args.comprehensive:
        print("Running comprehensive evaluation...")
        from siic.evaluation.comprehensive import main as comprehensive_main
        comprehensive_main()
        
    elif args.model:
        print(f"Evaluating {args.model} model...")
        from siic.evaluation.metrics import main as metrics_main
        # Pass model argument to metrics main
        sys.argv = ['evaluate.py', '--model', args.model]
        metrics_main()
        
    elif args.generate_report:
        print("Generating evaluation report...")
        from siic.evaluation.reports import main as reports_main
        reports_main()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 