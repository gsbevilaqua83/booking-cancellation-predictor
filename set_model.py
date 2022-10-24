import sys

# Script to set current model used by the api.
# It simply writes the model(run id) to a file read by the backend

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open('current_model', 'w') as f:
            f.write(sys.argv[1])