import numpy as np
#import mpl_toolkits.basemap as bm
from scipy.interpolate import interp1d
from scipy import stats
import datetime as dt

def stripnan(y,m,d):
    idx=np.isfinite(d) & np.isfinite(y) & np.isfinite(m)
    d=d[idx]
    m=m[idx]
    y=y[idx]
    return idx,y,m,d
    

def poly2detrend(years, months, co2):
    detrend_2=0*co2+np.nan
    [idx, years, months, co2]=stripnan(years,months,co2) 
    if any(idx):
        time=years+months/12.0
        fit_0=np.polyfit(time, co2, 2)
        detrend_0=co2-(fit_0[0]*time*time + fit_0[1]*time+fit_0[2])
        YEARS=np.arange(years[0],years[-1]+1,1)
        MONTHS=np.linspace(1,12,12, endpoint=True)
        mm=compute_mm(months, detrend_0)
        deseason_0=np.zeros(co2.shape)
        for imonth in range(len(MONTHS)):
            jmonth=np.nonzero(months==MONTHS[imonth])
            deseason_0[jmonth]=co2[jmonth]-mm[imonth]
        fit_1=np.polyfit(time, deseason_0, 2)
        trend_1=fit_1[0]*time*time + fit_1[1]*time+fit_1[2]
        #detrend_1=co2-(fit_1[0]*time*time + fit_1[1]*time+fit_1[2])
        detrend_1=co2-trend_1
        detrend_2[idx]=detrend_1
        trend_2[idx]=trend_1
        idx=[]
    return detrend_2,trend_2

def poly2trend(years, months, co2):
    detrend_2=0*co2+np.nan
    [idx, years, months, co2]=stripnan(years,months,co2) 
    if any(idx):
        time=years+months/12.0
        fit_0=np.polyfit(time, co2, 2)
        detrend_0=co2-(fit_0[0]*time*time + fit_0[1]*time+fit_0[2])
        YEARS=np.arange(years[0],years[-1]+1,1)
        MONTHS=np.linspace(1,12,12, endpoint=True)
        mm=compute_mm(months, detrend_0)
        deseason_0=np.zeros(co2.shape)
        for imonth in range(len(MONTHS)):
            jmonth=np.nonzero(months==MONTHS[imonth])
            deseason_0[jmonth]=co2[jmonth]-mm[imonth]
        fit_1=np.polyfit(time, deseason_0, 2)
        trend_1=fit_1[0]*time*time + fit_1[1]*time+fit_1[2]
        detrend_1=co2-trend_1
        detrend_2[idx]=detrend_1
        trend_2[idx]=trend_1
        idx=[]
    return trend_2

def fitexponential(years,months,co2):
    time=years+months/12.0
    lnco2=np.log(co2)
    fit_0=np.polyfit(time, lnco2, 1)
    coeff1=np.exp(fit_0[1])
    coeff2=fit_0[0]
    detrendedco2=co2-(coeff1*np.exp(time*coeff2))
    #detrend_0=co2-(coeff1*time+fit_0[1])
    #detrendedco2=np.exp(detrend_0)
    return detrendedco2

def poly3detrend(years, months, co2):
    time=years+months/12.0
    fit_0=np.polyfit(time, co2, 3)
    detrend_0=co2-(fit_0[0]*time*time*time + fit_0[1]*time*time+fit_0[2]*time+fit_0[3])
    YEARS=np.arange(years[0],years[-1]+1,1)
    MONTHS=np.linspace(1,12,12, endpoint=True)
    mm=compute_mm(months, detrend_0)
    deseason_0=np.zeros(co2.shape)
    for imonth in range(len(MONTHS)):
        jmonth=np.nonzero(months==MONTHS[imonth])
        deseason_0[jmonth]=co2[jmonth]-mm[imonth]
    fit_1=np.polyfit(time, deseason_0, 3)
    detrend_1=co2-(fit_1[0]*time*time*time + fit_1[1]*time*time+fit_1[2]*time+fit_1[3])
    return detrend_1

def poly2deseason(years, months, co2):
    time=years+months/12.0
    fit_0=np.polyfit(time, co2, 2)
    detrend_0=co2-(fit_0[0]*time*time + fit_0[1]*time+fit_0[2])
    YEARS=np.arange(years[0],years[-1]+1,1)
    MONTHS=np.linspace(1,12,12, endpoint=True)
    mm=compute_mm(months, detrend_0)
    deseason_0=np.zeros(co2.shape)
    for imonth in range(len(MONTHS)):
        jmonth=np.nonzero(months==MONTHS[imonth])
        deseason_0[jmonth]=co2[jmonth]-mm[imonth]
    return deseason_0

def poly2_standard(years, months, co2, years1, months1, co21):
    time1=years1+months1/12.0
    fit_0=np.polyfit(time1, co21, 2)
    time=years+months/12.0
    trendline=fit_0[0]*time1*time1+fit_0[1]*time1+fit_0[2]
    detrend_0=co21-trendline
    YEARS=np.arange(years1[0],years1[-1]+1,1)
    MONTHS=np.linspace(1,12,12, endpoint=True)
    mm=compute_mm(months1, detrend_0)
    deseason_0=np.zeros(co21.shape)
    for imonth in range(len(MONTHS)):
        jmonth=np.nonzero(months1==MONTHS[imonth])
        deseason_0[jmonth]=co21[jmonth]-mm[imonth]
    fit_1=np.polyfit(time1, deseason_0, 2)
    trendline=fit_1[0]*time*time+fit_1[1]*time+fit_1[2]
    detrend_1=co2-trendline
    return detrend_1

def poly2_deseason_standard(years, months, co2, years1, months1, co21):
    time1=years1+months1/12.0
    fit_0=np.polyfit(time1, co21, 2)
    time=years+months/12.0
    detrend_0=co2-(fit_0[0]*time*time + fit_0[1]*time+fit_0[2])
    YEARS=np.arange(years1[0],years1[-1]+1,1)
    MONTHS=np.linspace(1,12,12, endpoint=True)
    mm=compute_mm(months, detrend_0)
    deseason_0=np.zeros(co2.shape)
    for imonth in range(len(MONTHS)):
        jmonth=np.nonzero(months==MONTHS[imonth])
        deseason_0[jmonth]=co2[jmonth]-mm[imonth]
    return deseason_0

def compute_running_mm(years, months, co2):
    YEARS=np.arange(years[0],years[-1]+1,1)
    MONTHS=np.linspace(1,12,12, endpoint=True)    
    x=len(YEARS)
    mm=np.zeros((12*x,3), 'float')
    counter=0
    for iyr in range(len(YEARS)):
       for imonth in range(len(MONTHS)):
           jmonth=np.nonzero((months==MONTHS[imonth]) & (years==YEARS[iyr]))
           mm[counter,0]=YEARS[iyr]
           mm[counter,1]=MONTHS[imonth]
           mm[counter,2]=np.mean(co2[jmonth])
           counter+=1
    j=np.isnan(mm[:,2])
    mm[j,2]=0
    j2=np.nonzero(mm[:,2])
    mm=mm[j2,:]
    return mm

def compute_mm(months, co2):     
    MONTHS=np.linspace(1,12,12, endpoint=True)
    mm=np.zeros(MONTHS.shape)
    for imonth in range(len(MONTHS)):
        jmonth=np.nonzero(months==MONTHS[imonth])
        mm[imonth]=np.mean(co2[jmonth])
    return mm

def subtract_mean_sca(months, co2, mm):
    deseasonco2=np.zeros(co2.shape)
    for ipt in range(len(co2)):
       month_represented=months[ipt]
       deseasonco2[ipt]=co2[ipt]-mm[month_represented-1]
    return deseasonco2 

def compute_msdev(months, co2):     
    MONTHS=np.linspace(1,12,12, endpoint=True)
    msdev=np.zeros(MONTHS.shape)
    for imonth in range(len(MONTHS)):
        jmonth=np.nonzero(months==MONTHS[imonth])
        msdev[imonth]=np.std(co2[jmonth])
    return msdev

def compute_annual_mean(years, co2):
    YEARS=np.arange(years[0],years[-1]+1,1)
    annmean=np.zeros(YEARS.shape)
    for iyr in range(len(YEARS)):
        jyr=np.nonzero((years==YEARS[iyr]))
        annmean[iyr]=np.mean(co2[jyr])
    return annmean

def regrid(times, co2, newtimes):
    fin=interp1d(times, co2, 'cubic')
    newco2=fin(newtimes)
    return newco2

def regstats(time, co2):
    slope, intercept, rval, pval, see=stats.linregress(time, co2)
    mx=time.mean()
    sx2=((time-mx)**2).sum()
    sd_intercept=see*np.sqrt(1./len(time)+mx*mx/sx2)
    sd_slope=see*np.sqrt(1/sx2)
    return slope, intercept, pval, sd_slope, sd_intercept

def compute_daily_mean(day, hour, co2):
    days_with_data=np.unique(np.floor(day))
    dailymean=np.zeros(days_with_data.shape)
    for iday in range(len(days_with_data)):
        jday=np.nonzero(np.floor(day)==days_with_data[iday])
        dailymean[iday]=np.mean(co2[jday])
    return dailymean

def cdatenum(year, month, day):
    serialtime=np.zeros(year.shape)
    for ientry in range(len(year)):
        if month[ientry]==0:
            pytime=dt.datetime(int(year[ientry]), 1, 1)+dt.timedelta(days=int(day[ientry])-1)
        else:
            pytime=dt.datetime(int(year[ientry]), int(month[ientry]), int(day[ientry]))
        serialtime[ientry]=pytime.toordinal() 
    return serialtime

def cdatevec(serialtime):
   year=np.zeros(serialtime.shape)
   month=np.zeros(serialtime.shape)
   day=np.zeros(serialtime.shape)
   for ientry in range(len(serialtime)):
      if np.isnan(serialtime[ientry]):
          year[ientry]=np.nan
          month[ientry]=np.nan
          day[ientry]=np.nan
      else:
          timeobj=dt.date.fromordinal(int(serialtime[ientry]))
          year[ientry]=timeobj.year
          month[ientry]=timeobj.month
          day[ientry]=timeobj.day
   return year, month, day

def smooth(x,window_len, window='flat'):
#window = 'hanning'
    half_window_len=np.floor(window_len/2.0)
    if x.ndim != 1:
        raise ValueError( "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[half_window_len:-half_window_len]

def trend_by_month(years, months, data, month2plot):
    jmonth=np.nonzero(months==month2plot)[0]
    slope=np.polyfit(years[jmonth], data[jmonth],1)
    return slope[0]
 
