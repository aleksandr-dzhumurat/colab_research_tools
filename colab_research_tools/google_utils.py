from googleapiclient import discovery
from googleapiclient.errors import HttpError

import gspread_dataframe as gd
import gspread as gs
from google.auth import default

def get_drive_service():
    credentials, _ = default()

    drive_service = discovery.build('drive', 'v3', credentials=credentials)

    return drive_service

import datetime
import pandas as pd

def auth_google_from_colab():
    from google.colab import auth
    auth.authenticate_user()

def auth_google_sheets():
    import gspread
    from google.auth import default
    creds, _ = default()

    gc = gspread.authorize(creds)
    return gc

def read_google_sheet_link(gc, drive_link: str, worksheet: str = ''):
  """read_google_sheets(google_sheet_link, worksheet=worksheet_name)"""
  df = gc.open_by_url(drive_link)
  sheet_df = pd.DataFrame([])
  try:
    sheet_df = pd.DataFrame(df.worksheet(worksheet).get_all_records())
  except:
    print('Pls pass `worksheet` param: %s' % [i.title for i in df.worksheets()])

  return sheet_df

def find_on_drive(drive_service, file_name, is_folder = False):
  """
  find_on_drive('techcrunch_data', is_folder=True)
  find_on_drive('test_queries_2023-06-25_13:59')[:2]
  """
  response = []
  query = f"name = '{file_name}'"
  if is_folder:
    query += " and mimeType = 'application/vnd.google-apps.folder'"
  try:
    response = drive_service.files().list(q=query).execute()
    if 'files' in response:
        response = response['files']
  except HttpError as e:
      print("An error occurred while searching for the spreadsheet.\n", e)
  return response

def create_spreadsheet(file_name: str, parent_folder_id: str, drive_service):
  """
  """
  body = {
      'name': file_name,
      'mimeType': 'application/vnd.google-apps.spreadsheet',
      'parents': [parent_folder_id]
  }
  new_sheet = drive_service.files().create(body=body).execute()
  return new_sheet['id']

def get_or_create_spreadsheet(filename, folder_name, drive_service):
  drive_search_results = find_on_drive(drive_service, filename)
  spreadsheet_id = None
  if len(drive_search_results) > 0:
    res = sorted(drive_search_results, key=lambda x: x['id'])[0]
    spreadsheet_id = res['id']
    print('file %s exists: %s' % (filename, spreadsheet_id))
  else:
    print('file not found, creating')
    parent_dir_id = find_on_drive(drive_service, folder_name, is_folder=True)[0]['id']
    spreadsheet_id = create_spreadsheet(filename, parent_folder_id=parent_dir_id, drive_service=drive_service)
  return spreadsheet_id

def save_pandas(df, worksheet_id, credentials):
  gc = gs.authorize(credentials)
  wb = gc.open_by_key(worksheet_id)
  worksheet = wb.get_worksheet(0)
  data_for_sheets = [df.columns.to_list()] + df.values.tolist()
  print('Data saving...')
  worksheet.update('A1', data_for_sheets)
  print('Complited %d rows' % len(data_for_sheets))

def read_sheet_by_name(name, folder_name: str, gc, drive_service, worksheet=None):
  pocessed_tags_spreadsheet_id = get_or_create_spreadsheet(name, folder_name=folder_name, drive_service=drive_service)
  df = gc.open_by_key(pocessed_tags_spreadsheet_id)
  sheet_df = pd.DataFrame([])
  try:
    sheet_df = pd.DataFrame(df.worksheet(worksheet).get_all_records())
  except:
    worksheet_available = [i.title for i in df.worksheets()]
    print('Pls pass `worksheet` param: %s (now reading from %s)' % (worksheet_available, worksheet_available[0]))
    sheet_df = pd.DataFrame(df.worksheet(worksheet_available[0]).get_all_records())
  return sheet_df