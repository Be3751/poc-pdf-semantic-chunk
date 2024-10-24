import logging
from typing import Any, Iterator, List, Optional
import os
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob

logger = logging.getLogger(__name__)

from PIL import Image
import fitz  # PyMuPDF
import mimetypes
import tempfile

import base64
from mimetypes import guess_type
import concurrent.futures

from openai import AzureOpenAI
aoai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_api_key= os.getenv("AZURE_OPENAI_API_KEY")
aoai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def crop_image_from_image(image_path, page_number, bounding_box):
    """
    Crops an image based on a bounding box.

    :param image_path: Path to the image file.
    :param page_number: The page number of the image to crop (for TIFF format).
    :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
    :return: A cropped image.
    :rtype: PIL.Image.Image
    """
    with Image.open(image_path) as img:
        if img.format == "TIFF":
            # Open the TIFF image
            img.seek(page_number)
            img = img.copy()
            
        # The bounding box is expected to be in the format (left, upper, right, lower).
        cropped_image = img.crop(bounding_box)
        return cropped_image

def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    
    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    doc.close()

    return img

def crop_image_from_file(file_path, page_number, bounding_box):
    """
    Crop an image from a file.

    Args:
        file_path (str): The path to the file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        A PIL Image of the cropped area.
    """
    mime_type = mimetypes.guess_type(file_path)[0]
    
    return crop_image_from_pdf_page(file_path, page_number, bounding_box)


MAX_TOKENS = 2000

def understand_image_with_gptv(image_path, caption):
    """
    Generates a description for an image using the GPT-4V model.

    Parameters:
    - api_base (str): The base URL of the API.
    - api_key (str): The API key for authentication.
    - deployment_name (str): The name of the deployment.
    - api_version (str): The version of the API.
    - image_path (str): The path to the image file.
    - caption (str): The caption for the image.

    Returns:
    - img_description (str): The generated description for the image.
    """
    client = AzureOpenAI(
        api_key=aoai_api_key,  
        api_version=aoai_api_version,
        base_url=f"{aoai_api_base}/openai/deployments/{aoai_deployment_name}"
    )

    fewshot_user_message = """
画像の内容に沿った回答をしてください。
以下は画像の内容によって出力してほしい内容の例です。
---
フローチャートの場合:
まず全体の概要を述べ、その後各ステップを詳細に説明し、矢印の方向と流れについても触れてください。

グラフの場合:
まずグラフの種類を述べ、X軸とY軸のラベルを説明し、データポイントの概要とその傾向について詳しく教えてください。

図表の場合:
画像は図表です。この図表について説明してください。図の種類を含め、各部品の詳細と図表全体の目的や意図を解説してください。

表の場合:
行と列のラベル、各セルに含まれるデータの内容を述べ、最後に表全体のまとめをお願いします。

イラストの場合:
全体の概要を述べ、各部分の詳細な説明を行い、色や形についても詳しく述べてください。

写真の場合:
写真の全体の内容を述べ、背景や主要な要素について触れ、色味、明るさや構図についても詳しく説明してください。

スクリーンショットの場合:
全体の場面を述べ、各部分のUIコンポーネントについて詳細に説明し、操作や手順について具体的に語ってください。

マップの場合:
マップの種類と範囲、主要な場所やランドマークを述べ、経路や距離についても詳しく説明してください。
---
"""

    data_url = local_image_to_data_url(image_path)
    response = client.chat.completions.create(
                model=aoai_deployment_name,
                messages=[
                    { "role": "system", "content": "あなたは優秀なアシスタントです。" },
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": f"この画像を説明してください。 日本語で回答してください。(注:画像のキャプションがあります: {caption})" + fewshot_user_message + ":" if caption else "この画像を説明してください。日本語で回答してください。" + fewshot_user_message + ":"
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ] } 
                ]
            )
    img_description = response.choices[0].message.content
    return img_description, data_url



def update_figure_description(md_content, img_description, idx):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """

    # The substring you're looking for
    # start_substring = f"![](figures/{idx})"
    start_substring = f"<figure>"
    end_substring = "</figure>"
    new_string = f"<!-- FigureContent=\"{img_description}\" -->"
    
    new_md_content = md_content
    # Find the start and end indices of the part to replace
    start_index = md_content.find(start_substring)
    if start_index != -1:  # if start_substring is found
        start_index += len(start_substring)  # move the index to the end of start_substring
        end_index = md_content.find(end_substring, start_index)
        if end_index != -1:  # if end_substring is found
            # Replace the old string with the new string
            new_md_content = md_content[:start_index] + new_string + md_content[end_index:]
    
    return new_md_content

def process_figure(input_file_path, figure, idx, md_content, output_folder):
    figure_content = ""
    img_description = ""
    print(f"Figure #{idx} has the following spans: {figure.spans}")
    for i, span in enumerate(figure.spans):
        print(f"Span #{i}: {span}")
        figure_content += md_content[span.offset:span.offset + span.length]
    print(f"Original figure content in markdown: {figure_content}")

    # Note: figure bounding regions currently contain both the bounding region of figure caption and figure body
    if figure.caption:
        caption_region = figure.caption.bounding_regions
        print(f"\tCaption: {figure.caption.content}")
        print(f"\tCaption bounding region: {caption_region}")
        for region in figure.bounding_regions:
            if region not in caption_region:
                print(f"\tFigure body bounding regions: {region}")
                boundingbox = (
                        region.polygon[0],  # x0 (left)
                        region.polygon[1],  # y0 (top)
                        region.polygon[4],  # x1 (right)
                        region.polygon[5]   # y1 (bottom)
                    )
                print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")
                cropped_image = crop_image_from_file(input_file_path, region.page_number - 1, boundingbox) # page_number is 1-indexed

                # Get the base name of the file
                base_name = os.path.basename(input_file_path)
                # Remove the file extension
                file_name_without_extension = os.path.splitext(base_name)[0]

                output_file = f"{file_name_without_extension}_cropped_image_{idx}.png"
                cropped_image_filename = os.path.join(output_folder, output_file)
                
                # 一時ディレクトリを使用
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmp_cropped_image_filename = os.path.join(tmpdirname, output_file)
                    cropped_image.save(tmp_cropped_image_filename)
                    print(f"\tFigure {idx} cropped and saved as {tmp_cropped_image_filename}")
                    img_desc, image_url = understand_image_with_gptv(tmp_cropped_image_filename, figure.caption.content)
                
                img_description += img_desc
                print(f"\tDescription of figure {idx}: {img_description}")
    else:
        print("\tNo caption found for this figure.")
        for region in figure.bounding_regions:
            print(f"\tFigure body bounding regions: {region}")
            boundingbox = (
                    region.polygon[0],  # x0 (left)
                    region.polygon[1],  # y0, (top)
                    region.polygon[4],  # x1 (right)
                    region.polygon[5]   # y1 (bottom)
                )
            print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")

            cropped_image = crop_image_from_file(input_file_path, region.page_number - 1, boundingbox) # page_number is 1-indexed

            # Get the base name of the file
            base_name = os.path.basename(input_file_path)
            # Remove the file extension
            file_name_without_extension = os.path.splitext(base_name)[0]

            output_file = f"{file_name_without_extension}_cropped_image_{idx}.png"
            cropped_image_filename = os.path.join(output_folder, output_file)
            
            # 一時ディレクトリを使用
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_cropped_image_filename = os.path.join(tmpdirname, output_file)
                cropped_image.save(tmp_cropped_image_filename)
                print(f"\tFigure {idx} cropped and saved as {tmp_cropped_image_filename}")
                img_desc, image_url = understand_image_with_gptv(tmp_cropped_image_filename, "")
            
            img_description += img_desc
            print(f"\tDescription of figure {idx}: {img_description}")

    return idx, img_description, image_url

def include_figure_in_md(input_file_path, result, output_folder="/tmp"):
    md_content = result.content
    fig_metadata = {}

    if result.figures:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_figure, input_file_path, figure, idx, md_content, output_folder): idx for idx, figure in enumerate(result.figures)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    
                    index, img_description, image_url = future.result()
                    fig_metadata[index] = image_url
                    md_content = update_figure_description(md_content, img_description, index)
                except Exception as exc:
                    print(f"Figure {idx} generated an exception: {exc}")
    
    return md_content, fig_metadata

class AzureAIDocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Forms Recognizer)."""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
        analysis_features: Optional[List[str]] = None,
    ):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import DocumentAnalysisFeature
        from azure.core.credentials import AzureKeyCredential

        kwargs = {}
        if api_version is not None:
            kwargs["api_version"] = api_version

        if analysis_features is not None:
            _SUPPORTED_FEATURES = [
                DocumentAnalysisFeature.OCR_HIGH_RESOLUTION,
            ]

            analysis_features = [
                DocumentAnalysisFeature(feature) for feature in analysis_features
            ]
            if any(
                [feature not in _SUPPORTED_FEATURES for feature in analysis_features]
            ):
                logger.warning(
                    f"The current supported features are: "
                    f"{[f.value for f in _SUPPORTED_FEATURES]}. "
                    "Using other features may result in unexpected behavior."
                )

        self.client = DocumentIntelligenceClient(
            endpoint=api_endpoint,
            credential=AzureKeyCredential(api_key),
            headers={"x-ms-useragent": "langchain-parser/1.0.0"},
            features=analysis_features,
            **kwargs,
        )
        self.api_model = api_model
        self.mode = mode
        assert self.mode in ["single", "page", "markdown"]

    def _generate_docs_page(self, result: Any) -> Iterator[Document]:
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "page": p.page_number,
                },
            )
            yield d

    def _generate_docs_single(self, file_path: str, result: Any) -> Iterator[Document]:
        md_content, fig_metadata = include_figure_in_md(file_path, result)
        yield Document(page_content=md_content, metadata={"images": fig_metadata})

    def lazy_parse(self, file_path: str) -> Iterator[Document]:
        """Lazily parse the blob."""
        blob = Blob.from_path(file_path)
        with blob.as_bytes_io() as file_obj:
            
            poller = self.client.begin_analyze_document(
                self.api_model,
                file_obj,
                content_type="application/octet-stream",
                output_content_format="markdown" if self.mode == "markdown" else "text",
            )
            result = poller.result()

            if self.mode in ["single", "markdown"]:
                yield from self._generate_docs_single(file_path, result)
            elif self.mode in ["page"]:
                yield from self._generate_docs_page(result)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

    def parse_url(self, url: str) -> Iterator[Document]:
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        poller = self.client.begin_analyze_document(
            self.api_model,
            AnalyzeDocumentRequest(url_source=url),
            # content_type="application/octet-stream",
            output_content_format="markdown" if self.mode == "markdown" else "text",
        )
        result = poller.result()

        if self.mode in ["single", "markdown"]:
            yield from self._generate_docs_single(result)
        elif self.mode in ["page"]:
            yield from self._generate_docs_page(result)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

class AzureAIDocumentIntelligenceLoader(BaseLoader):
    """Loads a PDF with Azure Document Intelligence"""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        file_path: Optional[str] = None,
        url_path: Optional[str] = None,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
        *,
        analysis_features: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the object for file processing with Azure Document Intelligence
        (formerly Form Recognizer).

        This constructor initializes a AzureAIDocumentIntelligenceParser object to be
        used for parsing files using the Azure Document Intelligence API. The load
        method generates Documents whose content representations are determined by the
        mode parameter.

        Parameters:
        -----------
        api_endpoint: str
            The API endpoint to use for DocumentIntelligenceClient construction.
        api_key: str
            The API key to use for DocumentIntelligenceClient construction.
        file_path : Optional[str]
            The path to the file that needs to be loaded.
            Either file_path or url_path must be specified.
        url_path : Optional[str]
            The URL to the file that needs to be loaded.
            Either file_path or url_path must be specified.
        api_version: Optional[str]
            The API version for DocumentIntelligenceClient. Setting None to use
            the default value from `azure-ai-documentintelligence` package.
        api_model: str
            Unique document model name. Default value is "prebuilt-layout".
            Note that overriding this default value may result in unsupported
            behavior.
        mode: Optional[str]
            The type of content representation of the generated Documents.
            Use either "single", "page", or "markdown". Default value is "markdown".
        analysis_features: Optional[List[str]]
            List of optional analysis features, each feature should be passed
            as a str that conforms to the enum `DocumentAnalysisFeature` in
            `azure-ai-documentintelligence` package. Default value is None.

        Examples:
        ---------
        >>> obj = AzureAIDocumentIntelligenceLoader(
        ...     file_path="path/to/file",
        ...     api_endpoint="https://endpoint.azure.com",
        ...     api_key="APIKEY",
        ...     api_version="2023-10-31-preview",
        ...     api_model="prebuilt-layout",
        ...     mode="markdown"
        ... )
        """

        assert (
            file_path is not None or url_path is not None
        ), "file_path or url_path must be provided"
        self.file_path = file_path
        self.url_path = url_path

        self.parser = AzureAIDocumentIntelligenceParser(
            api_endpoint=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            api_model=api_model,
            mode=mode,
            analysis_features=analysis_features,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.file_path is not None:
            yield from self.parser.parse(self.file_path)
        else:
            yield from self.parser.parse_url(self.url_path)  # type: ignore[arg-type]