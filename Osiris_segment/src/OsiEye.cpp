/*******************************************************
* Open Source for Iris : OSIRIS
* Version : 4.0
* Date : 2011
* Author : Guillaume Sutra, Telecom SudParis, France
* License : BSD
********************************************************/


#include<opencv2/core/core.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include "cv.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;

#include <fstream>
#include <stdexcept>
#include "cv.h"
#include "OsiEye.h"
#include "OsiProcessings.h"

using namespace std ;

namespace osiris
{

    // CONSTRUCTORS & DESTRUCTORS
    /////////////////////////////

    OsiEye::OsiEye ( )
    {
        mpOriginalImage = 0 ;        
        mpSegmentedImage = 0 ;
        mpMask = 0 ;
        mpNormalizedImage = 0 ;
        mpNormalizedMask = 0 ;
        mpIrisCode = 0 ;
        mPupil.setCircle(0,0,0) ;
        mIris.setCircle(0,0,0) ;
    }

    OsiEye::~OsiEye ( )
    {
        cvReleaseImage(&mpOriginalImage) ;
        cvReleaseImage(&mpSegmentedImage) ;
        cvReleaseImage(&mpMask) ;
        cvReleaseImage(&mpNormalizedImage) ;
        cvReleaseImage(&mpNormalizedMask) ;
        cvReleaseImage(&mpIrisCode) ; 
		}





    // Functions for loading images and parameters
    //////////////////////////////////////////////

    void OsiEye::loadImage ( const string & rFilename , IplImage ** ppImage )
    {
        // :WARNING: ppImage is a pointer of pointer
        try
        {
            if ( *ppImage )
            {
                cvReleaseImage(ppImage) ;
            }

            *ppImage = cvLoadImage(rFilename.c_str(),0) ;
            if ( ! *ppImage )
            {
                cout << "Cannot load image : " << rFilename << endl ;
            }
        }
        catch ( exception & e )
        {
            cout << e.what() << endl ;
        }
    }



    void OsiEye::loadOriginalImage ( const string & rFilename )
    {
        loadImage(rFilename,&mpOriginalImage) ;
    }



    void OsiEye::loadMask ( const string & rFilename )
    {
        loadImage(rFilename,&mpMask) ;
    }



    void OsiEye::loadNormalizedImage ( const string & rFilename )
    {
        loadImage(rFilename,&mpNormalizedImage) ;
    }



    void OsiEye::loadNormalizedMask ( const string & rFilename )
    {
        loadImage(rFilename,&mpNormalizedMask) ;
    }



    void OsiEye::loadIrisCode ( const string & rFilename )
    {
        loadImage(rFilename,&mpIrisCode) ;
    }



    void OsiEye::loadParameters (const string & rFilename )
    {
        // Open the file
        ifstream file(rFilename.c_str(),ios::in) ;

        // If file is not opened
        if ( ! file )
        {
            throw runtime_error("Cannot load the parameters in " + rFilename) ;
        }
        try
        {            
            //int xp , yp , rp , xi , yi , ri ;
            //file >> xp ;
            //file >> yp ;
            //file >> rp ;
            //file >> xi ;
            //file >> yi ;
            //file >> ri ;
            //mPupil.setCircle(xp,yp,rp) ;
            //mIris.setCircle(xi,yi,ri) ;
			int nbp = 0 ;
			int nbi = 0 ;
			file >> nbp ;
			file >> nbi  ;
			mThetaCoarsePupil.resize(nbp, 0.0) ;
			mThetaCoarseIris.resize(nbi, 0.0) ;
			mCoarsePupilContour.resize(nbp, cvPoint(0,0)) ;
			mCoarseIrisContour.resize(nbi ,cvPoint(0,0)) ;
			//matrix.resize( num_of col , vector<double>( num_of_row , init_value ) );
			for (int i = 0 ; i < nbp ; i++)
			{
				file >> mCoarsePupilContour[i].x ;
				file >> mCoarsePupilContour[i].y ;
				file >> mThetaCoarsePupil[i] ;
			}
			for (int j = 0 ; j < nbi ; j++)
			{
				file >> mCoarseIrisContour[j].x ;
				file >> mCoarseIrisContour[j].y ;
				file >> mThetaCoarseIris[j] ;
			}
			
        }
        catch ( exception & e )
        {
            cout << e.what() << endl ;
            throw runtime_error("Error while loading parameters from " + rFilename) ;
        }

        // Close the file
        file.close() ;
    }






    // Functions for saving images and parameters
    /////////////////////////////////////////////

    void OsiEye::saveImage ( const string & rFilename , const IplImage * pImage )
    {
        return ;
        printf("save image %s %x\n",rFilename.c_str(),pImage);
        // :TODO: no exception here, but 2 error messages
        // 1. pImage does NOT exist => "image was neither comptued nor loaded"
        // 2. cvSaveImage returns <=0 => "rFilename = invalid for saving"
        if ( ! pImage )
        {
            throw runtime_error("Cannot save image " + rFilename + " because this image is not built") ;
        } 
imwrite(rFilename.c_str(),cvarrToMat(pImage));
/*
        if (!cvSaveImage(rFilename.c_str(),pImage) )
        {
            cout << "Cannot save image as " << rFilename << endl ;
        }
*/
    }



    void OsiEye::saveSegmentedImage ( const string & rFilename )
    {       
        return ; 
    }



    void OsiEye::saveMask ( const string & rFilename )
    {
        saveImage(rFilename,mpMask) ;
    }



    void OsiEye::saveNormalizedImage ( const string & rFilename )
    {
        saveImage(rFilename,mpNormalizedImage) ;
    }



    void OsiEye::saveNormalizedMask ( const string & rFilename )
    {
        saveImage(rFilename,mpNormalizedMask) ;
    }



    void OsiEye::saveIrisCode ( const string & rFilename )
    {
        saveImage(rFilename,mpIrisCode) ;
    }



    void OsiEye::saveParameters (const string & rFilename )
    {
        // Open the file
        ofstream file(rFilename.c_str(),ios::out) ;

        // If file is not opened
        if ( ! file )
        {
            throw runtime_error("Cannot save the parameters in " + rFilename) ;
        }
        
        try
        {
        //    file << mPupil.getCenter().x << " " ;
        //    file << mPupil.getCenter().y << " " ;
        //    file << mPupil.getRadius() << endl ;
        //    file << mIris.getCenter().x << " " ;
        //    file << mIris.getCenter().y << " " ;
        //    file << mIris.getRadius() << endl ;
			file << mCoarsePupilContour.size() << endl ;
			file << mCoarseIrisContour.size() << endl ;
			for (int i=0; i<(mCoarsePupilContour.size()); i++)
			{
				file << mCoarsePupilContour[i].x << " " ;
				file << mCoarsePupilContour[i].y << " " ;
				file << mThetaCoarsePupil[i] << " " ;
			}
			file << endl ;
			for (int j=0; j<(mCoarseIrisContour.size()); j++)
			{	
				file << mCoarseIrisContour[j].x << " " ;
				file << mCoarseIrisContour[j].y << " " ;
				file << mThetaCoarseIris[j] << " " ;
			}
        }
        catch ( exception & e )
        {
            cout << e.what() << endl ;
            throw runtime_error("Error while saving parameters in " + rFilename) ;
        }

        // Close the file
        file.close() ;
    }







    // Functions for processings
    ////////////////////////////



    void OsiEye::initMask ( )
    {
        if ( mpMask )
        {
            cvReleaseImage(&mpMask) ;
        }
        if ( ! mpOriginalImage )
        {
            throw runtime_error("Cannot initialize the mask because original image is not loaded") ;
        }
        mpMask = cvCreateImage(cvGetSize(mpOriginalImage),IPL_DEPTH_8U,1) ;
        cvSet(mpMask,cvScalar(255)) ;
    }

//void rotateBasedOnMask(lpl
    void OsiEye::segment ( int minIrisDiameter , int minPupilDiameter , int maxIrisDiameter , int maxPupilDiameter )
    { 
        if ( ! mpOriginalImage )
        {
            throw runtime_error("Cannot segment image because original image is not loaded") ;
        }

        // Initialize mask and segmented image
        mpMask = cvCreateImage(cvGetSize(mpOriginalImage),IPL_DEPTH_8U,1) ;
        mpSegmentedImage = cvCreateImage(cvGetSize(mpOriginalImage),IPL_DEPTH_8U,3) ;
        cvCvtColor(mpOriginalImage,mpSegmentedImage,CV_GRAY2BGR) ;
 
        // Processing functions
        OsiProcessings op ;
 
        // Segment the eye
        op.segment(mpOriginalImage,mpMask,mPupil,mIris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour,minIrisDiameter,minPupilDiameter,maxIrisDiameter,maxPupilDiameter) ;
 
        // Draw on segmented image
        IplImage * tmp = cvCloneImage(mpMask) ;
        cvZero(tmp) ;
        cvCircle(tmp,mIris.getCenter(),mIris.getRadius(),cvScalar(255),-1) ;
        cvCircle(tmp,mPupil.getCenter(),mPupil.getRadius(),cvScalar(0),-1) ;
        cvSub(tmp,mpMask,tmp) ;
        cvSet(mpSegmentedImage,cvScalar(0,0,255),tmp) ;
        cvCircle(mpSegmentedImage,mPupil.getCenter(),mPupil.getRadius(),cvScalar(0,255,0)) ;
        cvCircle(mpSegmentedImage,mIris.getCenter(),mIris.getRadius(),cvScalar(0,255,0)) ;
        
				int w=mIris.getRadius()*2; 
				//瞳孔中心坐标(不一定是同心圆)
				int pupil_c_x=mPupil.getCenter().x-mIris.getCenter().x+w/2;
				int pupil_c_y=mPupil.getCenter().y-mIris.getCenter().y+w/2;
				printf("[*]瞳孔中心和半径(%d,%d,%d),Iris中心和半径(%d,%d,%d)\n",
                    mPupil.getCenter().x,
                    mPupil.getCenter().y,
                    mPupil.getRadius(),
                    mIris.getCenter().x,
                    mIris.getCenter().y,
                    mIris.getRadius()
                    );
				 
        cvReleaseImage(&tmp) ;
    }



    void OsiEye::normalize ( int rWidthOfNormalizedIris , int rHeightOfNormalizedIris )
    {
        // Processing functions
        OsiProcessings op ;

        // For the image
        if ( ! mpOriginalImage )
        {
            throw runtime_error("Cannot normalize image because original image is not loaded") ;
        }
    
        mpNormalizedImage = cvCreateImage(cvSize(rWidthOfNormalizedIris,rHeightOfNormalizedIris),IPL_DEPTH_8U,1) ;

        if ( mThetaCoarsePupil.empty() || mThetaCoarseIris.empty() )
        {
            //throw runtime_error("Cannot normalize image because circles are not correctly computed") ;
			throw runtime_error("Cannot normalize image because contours are not correctly computed/loaded") ;
        }
        
        //op.normalize(mpOriginalImage,mpNormalizedImage,mPupil,mIris) ;
		op.normalizeFromContour(mpOriginalImage,mpNormalizedImage,mPupil,mIris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour) ;

        // For the mask
        if ( ! mpMask )
        {
            initMask() ;
        }

        mpNormalizedMask = cvCreateImage(cvSize(rWidthOfNormalizedIris,rHeightOfNormalizedIris),IPL_DEPTH_8U,1) ;
        
        //op.normalize(mpMask,mpNormalizedMask,mPupil,mIris) ;
		op.normalizeFromContour(mpMask,mpNormalizedMask,mPupil,mIris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour) ;
    }



    void OsiEye::encode ( const vector<CvMat*> & rGaborFilters )
    {
        if ( ! mpNormalizedImage )
        {
            throw runtime_error("Cannot encode because normalized image is not loaded") ;
        }

        // Create the image to store the iris code
        CvSize size = cvGetSize(mpNormalizedImage) ;
        mpIrisCode = cvCreateImage(cvSize(size.width,size.height*rGaborFilters.size()),IPL_DEPTH_8U,1) ;

        // Encode
        OsiProcessings op ;
        op.encode(mpNormalizedImage,mpIrisCode,rGaborFilters) ;
    }



    float OsiEye::match ( OsiEye & rEye , const CvMat * pApplicationPoints )
    { return 0;
    }



} // end of namespace
