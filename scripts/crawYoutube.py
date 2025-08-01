import csv

from googleapiclient.discovery import build

# Cấu hình
api_key = "AIzaSyAoQyi8E69GiZUqYIVsvhwkYP-wgfegVPI"
video_id = "f9P7_qWrf38"

youtube = build("youtube", "v3", developerKey=api_key)


def get_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token,
            )
            .execute()
        )

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                [
                    comment["authorDisplayName"],
                    comment["textDisplay"],
                    comment["likeCount"],
                    comment["publishedAt"],
                ]
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


# Gọi và lưu CSV
comments = get_comments(video_id)

with open("youtube_comments.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Comment", "Likes", "Time"])
    writer.writerows(comments)

print(f"Đã lưu {len(comments)} comment vào youtube_comments.csv")
