
import streamlit as st
from GD.gatingDesign import CalcArea, CalcRiser
from GD.saveFile import SaveFile
import numpy as np
import csv
import pandas as pd
from PIL import Image
from GD.modulus_dwg_image import *

st.header("Gating Design on Web")
ch = st.radio("รายการคำนวณ",("CalcGating","CalcModulus","CalcRiser","FileGating","FileRiser"))

if ch == "CalcGating":
    h = st.number_input("กรุณาเลือกตัวเลขความสูง",min_value=0,max_value=450,value=200,step=5)
    f = st.number_input("กรุณาเลือกตัวเลข f",min_value=0.0,max_value=0.9,value=0.4,step=0.1)
    q = st.number_input("กรุณาเลือกตัวเลข flowrate",min_value=0.0,max_value=10.0,value=1.0,step=0.1)
    p = st.number_input("กรุณาเลือกตัวเลขความสูงงานด้านบน",min_value=0,max_value=200,value=30,step=10)
    c = st.number_input("กรุณาเลือกตัวเลขความสูงงานทั้งหมด",min_value=0,max_value=400,value=100,step=10)
    th = st.number_input("กรุณาเลือกความหนา ingate",min_value=0.0,max_value=10.0,value=4.0,step=0.1)
    name = st.text_input("กรอกชื่อ runner_ratio_n/choke_ratio_n/ingate_ratio_n","runner_1.2_main1")

    fname = st.text_input("กรอกชื่อไฟล์ที่จะบันทึก(.csv)","out_gating.csv")

    st.text("ข้อมูลของคุณ คือ : " + f'h={h:.0f} f={f:0.2f} q={q:0.2f} p={p:0.1f} c={c:0.1f} name={name} gateTH={th:.1f}')
    c1 = CalcArea(h=h,f=f,q=q,p=p,c=c,name=name,gthickness=th)

    calc = st.button("คำนวณ gating")
    label_res = st.empty()
    label_res.text("คุณยังไม่ได้กดปุ่มcalc")
    save = st.button("บันทึกลงไฟล์")
    label_save = st.empty()
    label_save.text("คุณยังไม่ได้กดปุ่มsave")
    remove = st.button("ลบข้อมูลตัวล่าสุด")
    if calc:
        result = c1.save()
        label_res.text(f"ผลการคำนวณ: name:{result[5]} , area:{result[6]:.0f} mm2 , width:{result[7]:.0f} mm ,height:{result[9]:.1f} mm")

    if save:
        s = SaveFile()
        s.setdata(fname,CalcArea.data,'gating')
        s.save2csv()
        # savecsv_gating(fname,CalcArea.data)
        label_save.text("บันทึกผลคำนวณลงไฟล์ "+ fname + "เรียบร้อยแล้ว")
        with open(fname) as f:
            btn = st.download_button('Download File CSV',
                            data=f,
                            file_name=fname, 
                            mime='text/csv'
                            )  # Defaults to 'text/plain'
    if remove:
        if len(CalcArea.data) > 0 :
            c1.remove()
    dfm = pd.DataFrame(
                CalcArea.data,
                columns=SaveFile.gating_header)
    st.write(dfm)

if ch == "CalcModulus":
    uploaded_file = st.file_uploader("Choose Modulus input image (.jpeg) file")
    if uploaded_file is not None:
        #open image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #----------------------------------------------
        img_mod,img_dig = main_dwg_img(img)
        col1, col2 = st.columns(2)
        col1.image(img_mod, caption='ภาพ Modulus',channels="BGR")
        col2.image(img_dig, caption='ภาพ Diagram',channels="BGR")
       

if ch == "CalcRiser":

    mat = st.text_input("กรอกชื่อ mat FC25/FCD45","FCD45")
    cwt = st.number_input("กรุณาเลือกน้ำหนักชิ้นงาน",min_value=0.0,max_value=30.0,value=4.0,step=0.5)
    cmd = st.number_input("กรุณาเลือก Casting mod",min_value=0.0,max_value=2.0,value=1.0,step=0.1)
    nh = st.number_input("กรุณาเลือก ความหนา neck",min_value=0.0,max_value=30.0,value=0.0,step=0.5)
    cold = st.number_input("กรุณาเลือก ชนิด Riser Cold=0 , Hot=1",min_value=0,max_value=1,value=0,step=1)
    fname = st.text_input("กรอกชื่อไฟล์ที่จะบันทึก(.csv)","out_riser.csv")
    
    st.text("ข้อมูลของคุณ คือ : " + f'mat={mat} wt={cwt:.2f} mod={cmd} cold riser={cold} neck th={nh:.2f}')
    r1 = CalcRiser(mat,cwt,cmd,cold,nh)

    calc = st.button("คำนวณ Riser")
    label_res = st.empty()
    label_res.text("คุณยังไม่ได้กดปุ่มcalc")
    save = st.button("บันทึกลงไฟล์")
    label_save = st.empty()
    label_save.text("คุณยังไม่ได้กดปุ่มsave")
    # dwload = st.button("Dow load file")
    remove = st.button("ลบข้อมูลตัวล่าสุด")
    if calc:
        result = r1.save()
        label_res.text("ผลการคำนวณ: "+f'mat:{result[0]} , mod:{result[5]:.3f} cm , base:{result[9]:.0f} mm , top:{result[10]:.1f} mm, Height:{result[11]:.1f} mm, Weight:{result[12]:.3f} kg ,ratio:{result[13]:.1f} time')

    if save:
        s = SaveFile()
        s.setdata(fname,CalcRiser.data,'riser')
        s.save2csv()
        # savecsv_riser(fname,CalcRiser.data)
        label_save.text("บันทึกผลคำนวณลงไฟล์ "+ fname + "เรียบร้อยแล้ว")
        with open(fname) as f:
            btn = st.download_button('Download File CSV',
                                data=f,
                                file_name=fname, 
                                mime='text/csv'
                                )  # Defaults to 'text/plain'
    if remove:
        if len(CalcRiser.data) > 0 :
            r1.remove()

    dfr = pd.DataFrame(
                CalcRiser.data,
                columns=SaveFile.riser_header)
    st.write(dfr)

if ch == "FileGating":
    uploaded_file = st.file_uploader("Choose a gating input data (csv) file")
    if uploaded_file is not None:
            #read csv
        idfg=pd.read_csv(uploaded_file)
        CalcArea.data = []
        for i in range(len(idfg)):
            g2 = CalcArea(*list(idfg.loc[i]))
            g2.save()

        df2g = pd.DataFrame(
                CalcArea.data,
                columns=SaveFile.gating_header)
        st.write(df2g)
            #read xls or xlsx
            # df1=pd.read_excel(uploaded_file)
    fname = st.text_input("กรอกชื่อไฟล์ที่จะบันทึก(.csv)","out_gating.csv")
    save = st.button("บันทึกลงไฟล์")
    label_save = st.empty()
    label_save.text("คุณยังไม่ได้กดปุ่มsave")
    if save:
        s = SaveFile()
        s.setdata(fname,CalcArea.data,'gating')
        s.save2csv()
        # savecsv_gating(fname,CalcArea.data)
        label_save.text("บันทึกผลคำนวณลงไฟล์ "+ fname + "เรียบร้อยแล้ว")
        with open(fname) as f:
            btn = st.download_button('Download File CSV',
                                data=f,
                                file_name=fname, 
                                mime='text/csv'
                                )  # Defaults to 'text/plain'

if ch == "FileRiser":
    uploaded_file = st.file_uploader("Choose a riser input data (csv) file")
    if uploaded_file is not None:
            #read csv
        idfr=pd.read_csv(uploaded_file)
        CalcRiser.data = []
        for i in range(len(idfr)):
            r2 = CalcRiser(*list(idfr.loc[i]))
            r2.save()

        df2r = pd.DataFrame(
            CalcRiser.data,
            columns=SaveFile.riser_header)
        st.write(df2r)

    fname = st.text_input("กรอกชื่อไฟล์ที่จะบันทึก(.csv)","out_riser.csv")
    save = st.button("บันทึกลงไฟล์")
    label_save = st.empty()
    label_save.text("คุณยังไม่ได้กดปุ่มsave")
    if save:
        s = SaveFile()
        s.setdata(fname,CalcRiser.data,'riser')
        s.save2csv()
        # savecsv_riser(fname,CalcRiser.data)
        label_save.text("บันทึกผลคำนวณลงไฟล์ "+ fname + "เรียบร้อยแล้ว")
        with open(fname) as f:
            btn = st.download_button('Download File CSV',
                                data=f,
                                file_name=fname, 
                                mime='text/csv'
                                )  # Defaults to 'text/plain'
