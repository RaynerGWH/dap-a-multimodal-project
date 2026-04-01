ssh first, then:

chmod +x /install.sh
./install.sh

pkill -f jupyter && uvicorn server:app --host 0.0.0.0 --port 8000

# kill processes on port 8888
pkill -f "jupyter.*8888"