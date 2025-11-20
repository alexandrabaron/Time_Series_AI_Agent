#!/usr/bin/env python3
"""
TSci-Chat - Conversational Time Series Forecasting Agent
Main entry point for the conversational UI.

This replaces the old automated pipeline with an interactive Streamlit interface.
To run the old pipeline, use main_legacy.py instead.
"""

import os
import sys
from pathlib import Path

# Add the current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def run_conversational_ui():
    """
    Launch the Streamlit conversational UI.
    This is the new default way to use TSci.
    """
    import streamlit.web.cli as stcli
    
    # Path to the streamlit app
    streamlit_app_path = current_dir / "ui" / "streamlit_app.py"
    
    # Launch streamlit
    sys.argv = ["streamlit", "run", str(streamlit_app_path)]
    sys.exit(stcli.main())


def run_legacy_pipeline():
    """
    Run the old automated pipeline (legacy mode).
    Use this if you want the original behavior without UI.
    """
    from graph.agent_graph import TimeSeriesAgentGraph
    from config.default_config import DEFAULT_CONFIG
    import json
    from datetime import datetime
    import time
    
    print("=" * 60)
    print("TimeSeriesSciensist - LEGACY MODE")
    print("=" * 60)

    # 1. Create and customize config
    config = DEFAULT_CONFIG.copy()
    config["num_slices"] = 25
    config["input_length"] = 512
    config["horizon"] = 96
    config["data_path"] = "../dataset/ETTh1.csv"
    config["debug"] = False
    config["verbose"] = False
    config["date_column"] = "date"
    config["value_column"] = "OT"
    config["output_dir"] = "results"

    # 2. Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable before running.")
        sys.exit(1)

    # 3. Initialize orchestrator
    print("Initializing Time Series Agent Graph...")
    graph = TimeSeriesAgentGraph(config=config, model=config["llm_model"], debug=config["debug"])

    # 4. Run the workflow
    print("Running the time series agent workflow...")
    print(f"Configuration: {config['num_slices']} slices, {config['horizon']} horizon steps")
    print("=" * 60)
    
    start_time = time.time()
    results = graph.run()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"Average time per slice: {execution_time/config['num_slices']:.2f} seconds")

    # 5. Save results
    print("\n" + "=" * 60)
    print("Workflow execution completed!")
    print("=" * 60)
    
    results_dir = Path("results/reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    complete_report_path = results_dir / f"complete_time_series_report_{timestamp}.json"
    try:
        with open(complete_report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Complete results saved to: {complete_report_path}")
    except Exception as e:
        print(f"Error saving complete results: {e}")

    print("\nExperiment completed!")


if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--legacy":
        print("Running in LEGACY mode (automated pipeline)")
        run_legacy_pipeline()
    else:
        print("🚀 Launching TSci-Chat conversational interface...")
        print("💡 To use the old automated pipeline, run: python main.py --legacy")
        print()
        run_conversational_ui() 
