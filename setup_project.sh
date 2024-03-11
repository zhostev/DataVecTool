#!/bin/bash

# 工程名
PROJECT_NAME=DataVecTool

# 创建基础目录结构
echo "Creating project structure for $PROJECT_NAME..."
mkdir -p $PROJECT_NAME/dataveclib/loaders
mkdir -p $PROJECT_NAME/dataveclib/preprocessors
mkdir -p $PROJECT_NAME/dataveclib/vectorizers
mkdir -p $PROJECT_NAME/examples

# 创建__init__.py文件以使其成为Python包
touch $PROJECT_NAME/dataveclib/__init__.py
touch $PROJECT_NAME/dataveclib/loaders/__init__.py
touch $PROJECT_NAME/dataveclib/preprocessors/__init__.py
touch $PROJECT_NAME/dataveclib/vectorizers/__init__.py

# 创建空的模块文件
touch $PROJECT_NAME/dataveclib/loaders/table_loader.py
touch $PROJECT_NAME/dataveclib/loaders/text_loader.py
touch $PROJECT_NAME/dataveclib/loaders/image_loader.py
touch $PROJECT_NAME/dataveclib/loaders/video_loader.py

touch $PROJECT_NAME/dataveclib/preprocessors/table_preprocessor.py
touch $PROJECT_NAME/dataveclib/preprocessors/text_preprocessor.py
touch $PROJECT_NAME/dataveclib/preprocessors/image_preprocessor.py
touch $PROJECT_NAME/dataveclib/preprocessors/video_preprocessor.py

touch $PROJECT_NAME/dataveclib/vectorizers/table_vectorizer.py
touch $PROJECT_NAME/dataveclib/vectorizers/text_vectorizer.py
touch $PROJECT_NAME/dataveclib/vectorizers/image_vectorizer.py
touch $PROJECT_NAME/dataveclib/vectorizers/video_vectorizer.py

touch $PROJECT_NAME/dataveclib/utils.py

# 创建示例使用代码文件
echo "Creating example usage file..."
cat > $PROJECT_NAME/examples/example_usage.py <<EOF
# 示例代码展示如何使用这些组件
def main():
    print("DataVecTool example usage.")

if __name__ == "__main__":
    main()
EOF

# 创建README.md
echo "Creating README.md..."
cat > $PROJECT_NAME/README.md <<EOF
# $PROJECT_NAME

This project provides a flexible and modular framework for data loading, preprocessing, and vectorization to support machine learning model training.
EOF

# 创建requirements.txt
echo "Creating requirements.txt..."
cat > $PROJECT_NAME/requirements.txt <<EOF
pandas>=1.2.0
numpy>=1.19.5
scikit-learn>=0.24.1
torch>=1.7.1
torchvision>=0.8.2
Pillow>=8.1.0
opencv-python>=4.5.1.48
scipy>=1.6.0
EOF

echo "Project $PROJECT_NAME setup complete."
