import csv


class SaveFile:
    gating_header = ['Head', 'LossFactor', 'FlowRate', 'CastingHeight_P', 'TotalCasting_C', 'Name', 'Area',
                                 'Width','Wmin', 'Wmax', 'Height', 'RanoldNo.', 'StressX', 'StressY']
    riser_header = ['Material','CastingWt','CastingMod','ColdRiser',
                                'NeckMod','RiserMod',
                                'NeckW','NeckH','NeckL',
                                'RiserBase','RiserTop','RiserH','RiserWt','RiserFeed', 'ForceX', 'ForceY']
    
    def __init__(self) -> None:
        self.filename = ""
        self.data = []
        self.calctype = ""

    def setdata(self,filename,data,calctype):
        self.filename = filename
        self.data = data
        self.calctype = calctype

    def save2csv(self):
        header = SaveFile.gating_header if self.calctype == "gating" else SaveFile.riser_header
        # writing to csv file 
        with open(self.filename, 'w', encoding='UTF8', newline='') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            # writing the fields 
            csvwriter.writerow(header)
            csvwriter.writerows(self.data)
