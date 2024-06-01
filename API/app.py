from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()


@app.post('/file')
def _file_upload(
        my_file: UploadFile = File(...),
        first: str = Form(...),
        second: str = Form("default value  for second"),
):
    return {
        "name": my_file.filename,
        "first": first,
        "second": second
    }