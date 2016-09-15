import numpy as np
from mayavi import mlab

#import multi component variables 

def plot_Nd(var,ghostIn=[],wh=1,fig_size_x=800,fig_size_y=600,sliced=0,x_slice=-1,y_slice=-1,z_slice=-1):
    #wh=1 # 0: black background, 1: whithe background
    #first check the rank of input data
    var_shape=var.shape
    var_components=var_shape[-1]
    var_dim=len(var_shape)-1

    #set environment
    if ghostIn==[]:
        #no ghost input
        #axes numbering
        range_var=[0]*var_dim*2
        for i in range(var_dim):
            range_var[2*i]=0
            range_var[2*i+1]=1
        #axes boundary
        bounds_var_ax = [1]*(2*(len(var.shape)-1))
        for i in range((len(var.shape)-1)):
            bounds_var_ax[2*i]=1
            bounds_var_ax[2*i+1]=var.shape[i]
        #outline boundary
        bounds_var_ol = [1]*(2*(len(var.shape)-1))
        for i in range((len(var.shape)-1)):
            bounds_var_ol[2*i]=1
            bounds_var_ol[2*i+1]=var.shape[i]
        ax_line_width=1.0
        ol_line_width=1.0
 
    else:
        dghost=np.zeros(len(ghostIn))
        for i in range(len(dghost)):
            dghost[i]=float(ghostIn[i])/(var.shape[i]-2*ghostIn[i])
        #axes numbering
        range_var=[0]*var_dim*2
        for i in range(var_dim):
            range_var[2*i]=0-dghost[i]
            range_var[2*i+1]=1+dghost[i]
        #axes boundary
        bounds_var_ax = [1]*(2*(len(var.shape)-1))
        for i in range((len(var.shape)-1)):
            bounds_var_ax[2*i]=1
            bounds_var_ax[2*i+1]=var.shape[i]
        #outline boundary
        bounds_var_ol = [1]*(2*(len(var.shape)-1))
        for i in range((len(var.shape)-1)):
            bounds_var_ol[2*i]=1+ghostIn[i]
            bounds_var_ol[2*i+1]=var.shape[i]-ghostIn[i]
        ax_line_width=1.0
        ol_line_width=2.0

    if x_slice<0:#default slice middle
        x_slice_pt=float(bounds_var_ol[2*0+1])/2+float(bounds_var_ol[2*0])/2-1
    else:
        x_slice_pt=float(bounds_var_ol[2*0])+(float(bounds_var_ol[2*0+1])-float(bounds_var_ol[2*0]))*x_slice-1
    if y_slice<0:
        y_slice_pt=float(bounds_var_ol[2*1+1])/2+float(bounds_var_ol[2*1])/2-1
    else:
        y_slice_pt=float(bounds_var_ol[2*1])+(float(bounds_var_ol[2*1+1])-float(bounds_var_ol[2*1]))*y_slice-1
    if z_slice<0:
        z_slice_pt=float(bounds_var_ol[2*2+1])/2+float(bounds_var_ol[2*2])/2-1
    else:
        z_slice_pt=float(bounds_var_ol[2*2])+(float(bounds_var_ol[2*2+1])-float(bounds_var_ol[2*2]))*z_slice-1

        print bounds_var_ax
        print bounds_var_ol
        print x_slice_pt,x_slice
        print y_slice_pt,y_slice
        print z_slice_pt,z_slice

 
    #Start plotting   
    if var_dim==3:
        #try 3D plot using mayavi
        if var_components>1:
            #try vector field plot (x,y,z) = 0 2 -1
            fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
            src=mlab.pipeline.vector_field(var[:,:,:,0],var[:,:,:,2],-var[:,:,:,1])
            vh=mlab.pipeline.vectors(src,mask_points=4)
            #engine=mlab.get_engine()
            #vectors = engine.scenes[0].children[0].children[0].children[0]
            #vectors.glyph.glyph_source.glyph_source = vectors.glyph.glyph_source.glyph_dict['arrow_source']
            vh.glyph.glyph_source.glyph_source = vh.glyph.glyph_source.glyph_dict['arrow_source']
            vh.glyph.glyph.scale_factor = 1.0
        elif sliced==0 and (x_slice==-1 and y_slice==-1 and z_slice==-1):
            #try iso plot 
            fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
            ch=mlab.contour3d(var[:,:,:,0],contours=10,transparent=True,opacity=0.8)
            cb=mlab.colorbar(title='Magnetic field',orientation='vertical' )
            cb.title_text_property.color=(1-wh,1-wh,1-wh)
            cb.label_text_property.color=(1-wh,1-wh,1-wh)
        else:
            #try slice plot 
            fig=mlab.figure(bgcolor=(wh,wh,wh),size=(fig_size_x,fig_size_y))
            if sliced>0:
                sxh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='x_axes',slice_index=x_slice_pt)
                sxh.ipw.slice_position=x_slice_pt+1
                syh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='y_axes',slice_index=y_slice_pt)
                syh.ipw.slice_position=y_slice_pt+1
                szh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='z_axes',slice_index=z_slice_pt)
                szh.ipw.slice_position=z_slice_pt+1
            else:
                if x_slice>-1:
                    sxh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='x_axes',slice_index=x_slice_pt)
                    sxh.ipw.slice_position=x_slice_pt+1
                if y_slice>-1:
                    syh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='y_axes',slice_index=y_slice_pt)
                    syh.ipw.slice_position=y_slice_pt+1
                if z_slice>-1:
                    szh=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(var[:,:,:,0]),plane_orientation='z_axes',slice_index=z_slice_pt)
                    szh.ipw.slice_position=z_slice_pt+1

            cb=mlab.colorbar(title='Magnetic field',orientation='vertical' )
            cb.title_text_property.color=(1-wh,1-wh,1-wh)
            cb.label_text_property.color=(1-wh,1-wh,1-wh)


        ax = mlab.axes(nb_labels=5,ranges=range_var)
        ax.axes.property.color=(1-wh,1-wh,1-wh)
        ax.axes.property.line_width = ax_line_width
        ax.axes.bounds=(bounds_var_ax)
        ax.axes.axis_title_text_property.color = (1-wh,1-wh,1-wh)
        ax.axes.axis_label_text_property.color = (1-wh,1-wh,1-wh)
        ax.axes.label_format='%.2f'
        ol=mlab.outline(color=(1-wh,1-wh,1-wh),extent=bounds_var_ol)
        ol.actor.property.line_width = ol_line_width
        mlab.view(roll=0,azimuth=60,elevation=30,distance='auto')
        return fig

    return 






        
