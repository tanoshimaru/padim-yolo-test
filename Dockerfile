# ベースイメージの指定（Jetsonの場合はlatest-jetson-jetpack6，それ以外はlatestを使用）
# FROM ultralytics/ultralytics:latest-jetson-jetpack6
FROM ultralytics/ultralytics:latest

# ビルド引数の定義（docker-compose.ymlのargsから取得できない場合のデフォルト値）
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

# グループとユーザーがすでに存在する場合に備えて、既存のユーザーとグループを削除
RUN groupdel $USERNAME || true && userdel -r $USERNAME || true

# グループとユーザーを作成
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the working directory
WORKDIR /app

# ディレクトリの所有者を変更
RUN chown -R $USERNAME:$USERNAME /app

# Install cron and required packages, set timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y cron imagemagick tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo "Asia/Tokyo" > /etc/timezone

RUN pip install "setuptools<69" && \
    pip install anomalib dotenv einops FrEIA kornia lightning onnxslim open-clip-torch scikit-image tifffile timm && \
    pip install "numpy<2"&& \
    pip install -U setuptools

# ユーザーを切り替え
USER $USERNAME
