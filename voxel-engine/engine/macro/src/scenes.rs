use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

pub fn derive_scenes_impl(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    // Parse using syn::parse2 (works with proc_macro2::TokenStream)
    let DeriveInput { ident, data, .. } = match syn::parse2::<DeriveInput>(input2) {
        Ok(input) => input,
        Err(e) => return e.to_compile_error().into(),
    };
    let enum_name = &ident;

    // Validate that the input is an enum
    let enum_data = match &data {
        Data::Enum(data) => data,
        _ => {
            return syn::Error::new_spanned(&ident, "Scenes can only be derived for enums")
                .to_compile_error()
                .into();
        }
    };

    // Validate that all variants are tuple structs with exactly one field
    for variant in &enum_data.variants {
        match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                // Valid: tuple struct with one field
            }
            _ => {
                return syn::Error::new_spanned(
                    variant,
                    "Each variant must be a tuple struct with exactly one field that implements Scene",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // Generate match arms for each method
    let on_enter_arms = enum_data.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        quote! {
            #enum_name::#variant_name(scene) => scene.on_enter_systems()
        }
    });

    let on_update_arms = enum_data.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        quote! {
            #enum_name::#variant_name(scene) => scene.on_update_systems()
        }
    });

    let on_exit_arms = enum_data.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        quote! {
            #enum_name::#variant_name(scene) => scene.on_exit_systems()
        }
    });

    let expanded = quote! {
        impl voxel_engine::Scenes for #enum_name {
            fn on_enter_systems(&self) -> &[fn(&mut voxel_engine::World)] {
                match self {
                    #(#on_enter_arms,)*
                }
            }

            fn on_update_systems(&self) -> &[fn(&mut voxel_engine::World)] {
                match self {
                    #(#on_update_arms,)*
                }
            }

            fn on_exit_systems(&self) -> &[fn(&mut voxel_engine::World)] {
                match self {
                    #(#on_exit_arms,)*
                }
            }
        }
    };

    expanded.into()
}
