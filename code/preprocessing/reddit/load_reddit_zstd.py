"""
This script is copied and slightly changed from the PushShiftDumps 
Github: https://github.com/Watchful1/PushshiftDumps/tree/master.
It converts a zst file to a csv that contains the Reddit comments and posts. 
"""

import zstandard
import os
import json
import sys
import csv
from datetime import datetime
import logging.handlers
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

#list of subreddits
subreddits = [
'algotrading_comments',
'algotrading_submissions',
'Bogleheads_comments',
'Bogleheads_submissions',
'Daytrading_comments',
'Daytrading_submissions',
'dividends_comments',
'dividends_submissions',
'ETFs_comments',
'ETFs_submissions',
'ExpatFIRE_comments',
'ExpatFIRE_submissions',
'fatFIRE_comments',
'fatFIRE_submissions',
'financialindependence_comments',
'financialindependence_submissions',
'investing_comments',
'investing_discussion_comments',
'investing_discussion_submissions',
'investing_submissions',
'leanfire_comments',
'leanfire_submissions',
'options_comments',
'options_submissions',
'pennystocks_comments',
'pennystocks_submissions',
'realestateinvesting_comments',
'realestateinvesting_submissions',
'RealEstate_comments',
'RealEstate_submissions',
'SecurityAnalysis_comments',
'SecurityAnalysis_submissions',
'StockMarket_comments',
'StockMarket_submissions',
'stocks_comments',
'stocks_submissions',
'thewallstreet_comments',
'thewallstreet_submissions',
'ValueInvesting_comments',
'ValueInvesting_submissions',
'WallStreetbetsELITE_comments',
'WallStreetbetsELITE_submissions',
'Wallstreetbetsnew_comments',
'Wallstreetbetsnew_submissions',
'wallstreetbetsOGs_comments',
'wallstreetbetsOGs_submissions',
'wallstreetbets_comments',
'wallstreetbets_submissions',
'Wallstreetsilver_comments',
'Wallstreetsilver_submissions'
]


log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
		while True:
			chunk = read_and_decode(reader, 2**27, (2**29) * 2)
			if not chunk:
				break
			lines = (buffer + chunk).split("\n")

			for line in lines[:-1]:
				yield line, file_handle.tell()

			buffer = lines[-1]
		reader.close()


if __name__ == "__main__":

	#adding a loop to convert multiple subreddits
	for subreddit in subreddits:
		
		#adding paths for input and output
		fields = []
		input_file_path = r'datasets\reddit\reddit\subreddits23'
		output_file_path = r'datasets\reddit_csv'

		input_file_path = os.path.join(input_file_path, f"{subreddit}.zst")
		output_file_path = os.path.join(output_file_path, f"{subreddit}.csv")

		if len(sys.argv) >= 3:
			input_file_path = sys.argv[1]
			output_file_path = sys.argv[2]
			fields = sys.argv[3].split(",")

		is_submission = "submission" in input_file_path

		#retrieving the appropriate information
		if not len(fields):
			if is_submission:
				fields = ["title","created","text","id"]
			else:
				fields = ["created","body","link_id"]

		file_size = os.stat(input_file_path).st_size
		file_lines, bad_lines = 0, 0
		line, created = None, None
		output_file = open(output_file_path, "w", encoding='utf-8', newline="")
		writer = csv.writer(output_file)
		writer.writerow(fields)
		unique_keys = set()
		try:
			for line, file_bytes_processed in read_lines_zst(input_file_path):
				try:
					obj = json.loads(line)
					unique_keys.update(obj.keys())
					output_obj = []
					for field in fields:
						if field == "created":
							value = datetime.fromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d %H:%M")
						elif field == "link":
							if 'permalink' in obj:
								value = f"https://www.reddit.com{obj['permalink']}"
							else:
								value = f"https://www.reddit.com/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}/"
						elif field == "author":
							value = f"u/{obj['author']}"
						elif field == "text":
							if 'selftext' in obj:
								value = obj['selftext']
							else:
								value = ""
						else:
							value = obj[field]

						output_obj.append(str(value).encode("utf-8", errors='replace').decode())
					writer.writerow(output_obj)

					created = datetime.utcfromtimestamp(int(obj['created_utc']))
				except json.JSONDecodeError as err:
					bad_lines += 1
				file_lines += 1
				if file_lines % 100000 == 0:
					log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
		except KeyError as err:
			log.info(f"Object has no key: {err}")
			log.info(line)
		except Exception as err:
			log.info(err)
			log.info(line)

		output_file.close()
		log.info(f"Complete : {file_lines:,} : {bad_lines:,}")
