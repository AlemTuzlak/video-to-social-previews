## Introduction

This project is an automation script that let's you generate:
- A transcript from a video.
- Title, description and tags for the video based on the transcript.
- YouTube chapters based on the transcript.
- Social media posts based on the transcript.

## Setup
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the dependencies using pnpm:
   ```bash
   pnpm install
   ```
4. Create a `.env` file in the root directory and configure it (copy .env.example as a starting point)
5. Create a whisper docker container locally by running the following:
```bash
docker build -t whispercpp-api .
docker run -d --name whisper --restart unless-stopped -p 9000:9000 -v "$PWD/models:/models" whispercpp-api
```
6. (Optional if you're not using openai) Create a locally running Ollama server and pull a compatible model (e.g. `gpt-4o-mini`):
   ```bash
   ollama pull gpt-4o-mini
   ```
6. add your video file to the `inbox` folder.
7. Run the script with:
   ```bash
   npm run generate ./inbox/your-video-file.mp4
   # Or to just generate the transcript:
   npm run generate ./inbox/your-video-file.mp4 --transcript-only
   ```