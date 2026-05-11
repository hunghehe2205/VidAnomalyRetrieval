import json
import requests
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Batch summarize videos")
    parser.add_argument("--input", default="input_videos.json", help="Input JSON file containing a list of video objects")
    parser.add_argument("--output", default="output_summaries.jsonl", help="Output JSONL file")
    parser.add_argument("--url", default="http://localhost:8080/summarize", help="API URL")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            videos = json.load(f)
    except Exception as e:
        print(f"Error loading {args.input}: {e}")
        sys.exit(1)

    processed_videos = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "video" in data:
                            processed_videos.add(data["video"])
                    except:
                        pass

    with open(args.output, "a", encoding="utf-8") as f:
        for i, video_data in enumerate(videos):
            video_name = video_data.get("video", f"unknown_{i}")
            if video_name in processed_videos:
                print(f"Skipping already processed video: {video_name}")
                continue
                
            print(f"Processing: {video_name} ({i+1}/{len(videos)})")
            try:
                response = requests.post(args.url, json=video_data, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    print(f"Success: {video_name}")
                else:
                    print(f"Failed: {video_name} - HTTP {response.status_code}")
                    error_data = {
                        "video": video_name,
                        "error": f"HTTP {response.status_code}",
                        "details": response.text
                    }
                    f.write(json.dumps(error_data) + "\n")
                    f.flush()
            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                error_data = {
                    "video": video_name,
                    "error": "Request Exception",
                    "details": str(e)
                }
                f.write(json.dumps(error_data) + "\n")
                f.flush()

if __name__ == "__main__":
    main()
