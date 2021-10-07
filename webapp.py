import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import datetime

CURRENT_YEAR = datetime.datetime.today().year

# image = Image.open('Untitled.png')
# st.image(image, use_column_width='always')

st.title('Khám sàng lọc bệnh tiểu đường')

full_name = st.text_input('Nhập họ và tên đầy đủ:')
id_number = st.text_input('Nhập số CMND / CCCD / Hộ chiếu:')

year_born = st.number_input(
    'Nhập năm sinh:',
    min_value=1900,
    max_value=CURRENT_YEAR-6,
    format='%d'
)

age = CURRENT_YEAR - year_born

gender = st.radio(
    'Chọn giới tính:',
    ('Nam', 'Nữ')
)

polyuria = st.radio(
    'Bạn có bị Đa niệu không? (Đi tiểu nhiều hơn 2 lít / ngày)',
    ('Có', 'Không')
)

polydipsia = st.radio(
    'Bạn có cảm thấy khát mặc dù vẫn uống nhiều nước không?',
    ('Có', 'Không')
)

weight_loss = st.radio(
    'Bạn có bị sút cân đột ngột không?',
    ('Có', 'Không')
)

weakness = st.radio(
    'Bạn có thường xuyên bị mệt mỏi không?',
    ('Có', 'Không')
)

polyphagia = st.radio(
    'Bạn có thường xuyên thèm ăn không?',
    ('Có', 'Không')
)

genital_thrush = st.radio(
    'Bạn có bị tưa miệng không?',
    ('Có', 'Không')
)

blurring = st.radio(
    'Bạn có bị mờ mắt không?',
    ('Có', 'Không')
)

itching = st.radio(
    'Bạn có thường xuyên bị ngứa không?',
    ('Có', 'Không')
)

irritability = st.radio(
    'Bạn có thường xuyên cáu gắt không?',
    ('Có', 'Không')
)

delayed_healing = st.radio(
    'Vết thương của bạn có lâu lành hay không?',
    ('Có', 'Không')
)

partial_paresis = st.radio(
    'Bạn có bị liệt một phần hay không?',
    ('Có', 'Không')
)

muscle_stiffness = st.radio(
    'Bạn có bị cứng cơ hay không?',
    ('Có', 'Không')
)

alopecia = st.radio(
    'Bạn có thường xuyên bị rụng tóc từng mảng hay không?',
    ('Có', 'Không')
)

obesity = st.radio(
    'Bạn có bị bệnh béo phì hay không?',
    ('Có', 'Không')
)

model = pickle.load(open('model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

if gender == 'Nam':
    gender = 1
else:
    gender = 0

age = scaler.transform([[age]])[0][0]

values = [age, gender, polyuria, polydipsia, weight_loss, weakness, polyphagia,
          genital_thrush, blurring, itching, irritability, delayed_healing,
          partial_paresis, muscle_stiffness, alopecia, obesity]

idx = 2
while idx < len(values):
    if values[idx] == 'Có':
        values[idx] = 1
    else:
        values[idx] = 0
    idx += 1

np_values = np.array(values).reshape(1, -1)

submit = st.button('Xong')

from fpdf import FPDF
import barcode
from barcode import EAN13
from barcode.writer import ImageWriter

if submit:
    y_pred = model.predict_proba(np_values)
    if y_pred[0][0] < 0.638:
        st.text('Bạn có nguy cơ cao bị mắc bệnh Tiểu đường !')

        # pdf = FPDF()
        # pdf.add_page()
        # pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        # pdf.set_font('DejaVu', '', 14)
        #
        # pdf.cell(200, 10, txt='BỘ Y TẾ',
        #          ln=1, align='C')
        #
        # pdf.cell(200, 10, txt ='BỆNH VIỆN DIABETES',
        #          ln=2, align='C')
        #
        # pdf.cell(200, 10, txt ='',
        #          ln=3, align='C')
        # pdf.cell(200, 10, txt ='',
        #          ln=4, align='C')
        # pdf.set_font('DejaVu', '', 18)
        # pdf.cell(200, 10, txt ='GIẤY HẸN KHÁM BỆNH',
        #          ln=5, align='C')
        #
        # ean = barcode.get('ean13', '123456789102', writer=ImageWriter())
        # filename = ean.save('ean13')
        # pdf.image(filename, x=1, y=3, w=70, h=20)
        #
        # pdf.cell(200, 10, txt ='',
        #          ln=6, align='C')
        # pdf.cell(200, 10, txt ='',
        #          ln=7, align='C')
        #
        # pdf.set_font('DejaVu', '', 12)
        # pdf.cell(200, 10, txt =f'Họ và tên: {full_name}',
        #          ln=8, align='L')
        # pdf.cell(200, 10, txt =f'Số CMND / CCCD / Hộ chiếu: {id_number}',
        #          ln=9, align='L')
        # pdf.cell(200, 10, txt =f'Ngày giờ tới khám: 15:00 ngày 20/11',
        #          ln=10, align='L')
        # pdf.cell(200, 10, txt =f'Chẩn đoán: Mắc bệnh Tiểu đường',
        #          ln=11, align='L')
        #
        #
        #
        # pdf.output("test.pdf")

    else:
        st.text('Bạn không có nguy cơ bị mắc bệnh Tiểu đường !')
