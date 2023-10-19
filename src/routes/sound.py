import asyncio
import io
from tempfile import NamedTemporaryFile
from typing import *

import numpy as np
from fastapi import APIRouter, Depends, File, Response, UploadFile
from pydub import AudioSegment
from scipy.io import wavfile

from ..apis import *
from ..client import *
from ..schemas import *
from ..utils import async_io

BUCKET_NAME = os.environ["AWS_S3_BUCKET"]
builder = QueryBuilder()
app = APIRouter(prefix="/api", tags=["sound"])


@async_io
def process_audio(file: bytes) -> AudioSegment:
	"""Preprocesses an audio file."""
	with NamedTemporaryFile() as temp:
		temp.write(file)
		temp.seek(0)
		audio = AudioSegment.from_file(temp.name)  # type: ignore
		if audio.channels == 2:
			audio = audio.set_channels(1)
		return audio


@app.post("/downsample/frequency")	# UI Done
async def downsample_to_frequency(
	file: UploadFile = File(...), sample_rate: int = 1536
) -> List[float]:
	"""Downsamples an audio file to the given sample rate."""
	binary_audio = await file.read()
	audio = await process_audio(binary_audio)
	audio.export(binary_audio, format="wav")
	wav_data = io.BytesIO(binary_audio)
	wav_data.seek(0)
	_, audio_sample = wavfile.read(wav_data)
	freq_domain = np.fft.fft(audio_sample)
	freq_domain = np.fft.fftshift(freq_domain)
	freq_domain = freq_domain[len(freq_domain) // 2 :]
	step_size = len(freq_domain) // sample_rate
	subsampled_audio = freq_domain[::step_size][:sample_rate]
	normalized_audio = subsampled_audio / np.linalg.norm(subsampled_audio)
	return normalized_audio.tolist()


@app.post("/downsample/time")
async def downsample_to_time(
	file: UploadFile = File(...), sample_rate: int = 1536
) -> List[float]:
	"""Downsamples an audio file to the given sample rate."""
	binary_audio = await file.read()
	audio: AudioSegment = await process_audio(binary_audio)
	wav_data = io.BytesIO()
	audio.export(wav_data, format="wav") # type: ignore
	wav_data.seek(0)
	_, audio_sample = wavfile.read(wav_data)
	step_size = len(audio_sample) // sample_rate
	subsampled_audio = audio_sample[::step_size][:sample_rate]
	normalized_audio = subsampled_audio / np.linalg.norm(subsampled_audio)
	return normalized_audio.tolist()

@app.post("/upload") # UI Done
async def upload_endpoint_post(user:str,key:str,file: UploadFile = File(...)):
	"""Uploads an audio file to the database."""
	client = S3Client()
	await client.put_object(key=key,body=await file.read())
	return await AudioUpload(user=user,key=key).save()

@app.get("/upload") # UI Done
async def upload_endpoint_get(user:str):
	"""Uploads an audio file to the database."""
	audio_uploads = await AudioUpload.find_many(user=user)
	client = S3Client()
	urls:list[str] = await asyncio.gather(*[client.get_object(key=upload.key) for upload in audio_uploads])
	users:list[FreeSoundUser] = await asyncio.gather(*[FreeSoundUser.get(upload.user) for upload in audio_uploads])
	return [{"url":url,"user":user.username,"ref":upload.ref} for url,user,upload in zip(urls,users,audio_uploads)]
		
@app.delete("/upload") # UI Done
async def upload_endpoint_delete(ref:str):
	"""Uploads an audio file to the database."""
	client = S3Client()
	audio_upload = await AudioUpload.get(ref=ref)
	await client.delete_object(key=audio_upload.key)
	return await AudioUpload.delete(ref=ref)

@app.get("/search") # UI Done
async def search_endpoint(token: str, query: str, filter: Filter, sort: Sort):
	"""Searches for audio files in the FreeSound database."""
	client = from_headers(headers={"Authorization": f"Bearer {token}"})
	return await client.search_sound(query=query, filter=filter, sort=sort)

@app.get("/fetch") # UI Done
async def fetch_endpoint(token: str, id: int):
	"""Fetches a sound from the database."""
	client = from_headers(headers={"Authorization": f"Bearer {token}"})
	return await client.get_sound(id=id)

@app.post("/download") # UI Done
async def download_endpoint(token: str, id: int):
	"""Downloads a sound from the database."""
	client = from_headers(headers={"Authorization": f"Bearer {token}"})
	data = await client.download_sound(id=id)
	return Response(content=data, media_type="audio/wav")

@app.get("/query")
async def similarity_search(user:str,vector: List[float], k: int = 10):
	"""Performs a similarity search on the database."""
	client = VectorClient()
	expr = (builder("user")==user).query
	response = await client.query(expr=expr,vector=vector,topK=k)
	return [r.metadata for r in response.matches]

@app.post("/upsert")
async def upsert_records(audiofile:AudioFile):
	"""Upserts a record into the database."""
	client = VectorClient()
	vector = audiofile.vector
	metadata = {k:v for k,v in audiofile.dict().items() if k!="vector"}
	embeddings = Embedding(values=vector,metadata=metadata)
	return await client.upsert(embeddings=[embeddings])