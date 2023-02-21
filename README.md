# TimeSeriesTraitsCalculator
A GUI tool to calculate Phenotyping traits data (Height and NDVI) .

Ececutable file: TimeSeriesTraitsCalculator.py

Tool Information:

Expected format for Height Data collected from Drone):
            columns: 'Plot', 'ComonVar', 'Nrate', 'Pesticide', 'Sowing date', 'Replicate', 'Row', 'Column', 'Ht_Prcnt99 yyyy-mm-dd'/
            'NDVI_Mean yyyy-mm-dd'

        Expected format for NDVI data collected using different methods):  an excel file should contain aleast these columns: 
            'Plot ID', 'Sowing date', 'Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167', 
            and dates columns with format: 'NDVI_UASInd YYYY-MM-DD sample_1' for Sample 1,
                                            'NDVI_UASInd YYYY-MM-DD sample_2' for Sample 2,
                                            'NDVI_UASInd YYYY-MM-DD sample_3' for Sample 3, and
                                            'NDVI_HHInd YYYY-MM-DD' for tec5 data.

        How to use the tool:
            1. Select the type of input data (height/ NDVI).
            2. Select the file where data is present.
            3. Select the location where you want to save the output file.
                (for option 2 and 3, you can't change location of file from the entry, you need to select it fron the 'Open Files' buttons)
            4. Select atleast one variable to fit data to (Date/ thermal time/ Days after Sowing).
            5. (Optional) Either remove the outliers or not, you will get better idea about how model behaves after 
                and before removing outliers by looking at output images.
            6. (Optional) If you want to get the corrensponding images to the output, select the check button and
                 give the location of folder where you want to save the images.
            7. (Optional) If you want to extract specific traits, select the kind of traits from the list and enter values in the entry boxes.

Processing Information:

For Height Data collected by Drone:

            m1/ model1: 
                Two different equations are used. One is Gompertz equation for growth phase and another is Linear equation
                to detect declining phase. The point of intersection of these two equations define the point of max. height (but this is
                true in case only when linear line is having positive slope). When the lines from two equations don't intersect, the
                first point of linear line is used as point of max. height.
            m2/ model2: 
                A single equation is formed by adding the two equations (Gompertz and Linear). Sometimes it find it hard to
                find optimal parameters. But reason of choosing this model is to find consistency that is missing in  model1. Model2 is
                performing better with thermal time data than with height data for existing datasets.
                negative points are replaced with 0.

            Comparison between two models:
                The two models are compared based on r2 score measure. If model2 has higher R2 score, then the traits are recorded from
                model2 as well.

            Output:
                The output consist of an Excel sheet, that will contain all traits data corresponding to the plots in columns. The data
                will be based on the options that the user has chosen.
                Asterisk(*) values are values that may not truly represent the trait.

            Images:
                blue lines represent model 1
                yellow line represent model 2
                green dots are points that are removed (outliers)
                small black star represent intersection point

        For NDVI Data collected by Drone:
            Only plots with SFP tratment are taken.

            Model: A double logistic curve is used for the curve representing NDVI for each row. 
            The model is generally underestimating the height.

            Images:
                Blue line represents the double logistic curve
                Green dots are the points that are removed (outliers)

        Model may not behave for:
            the plots where Nrate is zero,
            the plots where pesticide is RFP,
            the recorded points that are very dodgy and for
            the plots with insufficient data.
