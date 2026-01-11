1. Download files all in one folder 
2. Open terminal and navigate to the folder (cd /path/to/AnimalClassification) 
3. Activate your Python environment (conda activate Your_Python_Env, or if using venv: source venv/bin/activate)
4. Install dependencies if not already installed (pip install torch torchvision fastapi uvicorn pillow python-multipart) 
5. Start the API (uvicorn api:app --reload) 
6. Open API docs in browser, go to: http://127.0.0.1:8000 (or whatever the address is listed as in your terminal)
7. Test your model: 
	- click POST /predict -> "Try it out"
	- click Choose File -> upload a .jpg image of an animal
	- click Execute -> you'll see the predicted animal name